from fastapi import FastAPI
import asyncio
import os
from pathlib import Path
import logging
import hashlib
import json
import pandas as pd
from dotenv import load_dotenv

from utils import get_file_hash, get_cached_result, save_to_cache, optimize_image, validate_query_result
from textract import process_with_textract, extract_text_from_textract_response
from llm_text import process_with_llm_text
from llm_image import process_with_llm_image
from validation import CSVValidator

# Load environment variables from .env file
load_dotenv()

# Set up more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Define your custom queries for the adapter
QUERIES = [
    {"Text": "What is the chassis number?"},
    {"Text": "What is the engine number?"},
    {"Text": "What is the make?"},
    {"Text": "What is the model?"},
    {"Text": "What is the color?"},
    {"Text": "What is the customer name?"}
]

# Result cache
result_cache = {}

async def process_file(file_path: str, csv_validator=None):
    """Process a single invoice file with 3-attempt approach and CSV validation."""
    try:
        # Check cache first
        file_hash = get_file_hash(file_path)
        cached_result = await get_cached_result(file_hash, result_cache)
        if cached_result:
            logger.info(f"Using cached result for {file_path}")
            return cached_result

        # Initialize results dictionary with "Not Found" for ALL queries
        results = {query["Text"]: "Not Found" for query in QUERIES}
        confidence_scores = {query["Text"]: 0.0 for query in QUERIES}
        sources = {query["Text"]: "None" for query in QUERIES}
        validation_results = {query["Text"]: False for query in QUERIES}
        raw_extracted_text = ""
        
        # FIRST ATTEMPT: Textract with adapter
        logger.info(f"ATTEMPT 1: Processing {file_path} with Textract adapter")
        
        # Optimize image before sending to Textract
        optimized_image_bytes = optimize_image(file_path)
        
        # Process with Textract and get results
        textract_response, textract_results, textract_confidence, textract_validation = await process_with_textract(file_path, optimized_image_bytes, QUERIES)
        
        # Extract raw text for potential second attempt
        raw_extracted_text = extract_text_from_textract_response(textract_response)
        logger.info(f"Extracted {len(raw_extracted_text)} characters of raw text from document")
        
        # Update results with Textract findings
        for query_text, result_text in textract_results.items():
            results[query_text] = result_text
            confidence_scores[query_text] = textract_confidence.get(query_text, 0.0)
            sources[query_text] = "Textract (Attempt 1)"
            validation_results[query_text] = textract_validation.get(query_text, False)
        
        # Determine queries for second attempt
        queries_for_second_attempt = []
        
        # Add queries with failed validation or low confidence
        for query_text, result_text in results.items():
            if not validation_results.get(query_text, False) or confidence_scores.get(query_text, 0) < 85.0:
                queries_for_second_attempt.append({"Text": query_text})
                logger.info(f"Query '{query_text}' with value '{result_text}' added to second attempt: validation={validation_results.get(query_text, False)}, confidence={confidence_scores.get(query_text, 0)}")
        
        # SECOND ATTEMPT: OpenAI with extracted text (more token efficient)
        if queries_for_second_attempt and raw_extracted_text:
            logger.info(f"ATTEMPT 2: Processing {len(queries_for_second_attempt)} queries with OpenAI using extracted text")
            
            llm_results = await process_with_llm_text(raw_extracted_text, queries_for_second_attempt, 2)
            
            # Process the results
            for query_text, result in llm_results.items():
                if isinstance(result, dict):
                    value = result.get("value", "Not Found")
                    
                    # Only update if the result is valid or better than what we had
                    if value != "Not Found" and validate_query_result(query_text, value):
                        results[query_text] = value
                        confidence_scores[query_text] = result.get("confidence", 50.0)  # Default to medium confidence
                        sources[query_text] = "OpenAI Text (Attempt 2)"
                        validation_results[query_text] = True
                        logger.info(f"Updated with text-based result for '{query_text}': Value='{value}', Valid=True")
        
        # Determine queries for third attempt
        queries_for_third_attempt = []
        
        # Add any remaining "Not Found" or failed validation items
        for query_text, result_text in results.items():
            if result_text == "Not Found" or not validation_results.get(query_text, False):
                queries_for_third_attempt.append({"Text": query_text})
                logger.info(f"Query '{query_text}' with value '{result_text}' added to third attempt: validation={validation_results.get(query_text, False)}")
        
        # THIRD ATTEMPT: OpenAI with image for remaining issues
        if queries_for_third_attempt:
            logger.info(f"ATTEMPT 3: Processing {len(queries_for_third_attempt)} queries with OpenAI using image")
            
            llm_results = await process_with_llm_image(file_path, queries_for_third_attempt, 3)
            
            # Process the results
            for query_text, result in llm_results.items():
                if isinstance(result, dict):
                    value = result.get("value", "Not Found")
                    
                    # Only update if the result is valid or better than what we had
                    if value != "Not Found" and validate_query_result(query_text, value):
                        results[query_text] = value
                        confidence_scores[query_text] = result.get("confidence", 50.0)
                        sources[query_text] = "OpenAI Image (Attempt 3)"
                        validation_results[query_text] = True
                        logger.info(f"Updated with image-based result for '{query_text}': Value='{value}', Valid=True")
                    elif results[query_text] == "Not Found":
                        # Explicitly mark the source as Attempt 3 even if value is still "Not Found"
                        sources[query_text] = "OpenAI Image (Attempt 3)"
        
        # Validate against CSV if provided
        csv_validation_results = {}
        if csv_validator is not None:
            logger.info(f"Validating results against CSV database")
            try:
                csv_validation_results = await csv_validator.validate_against_csv(results)
                
                # Update validation results
                for query_text, is_valid in csv_validation_results.items():
                    validation_results[query_text] = is_valid
                    
                    # Add CSV as source for fully validated items
                    if is_valid:
                        sources[query_text] += " + CSV Validated"
            except Exception as e:
                logger.error(f"Error during CSV validation: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Prepare final result
        final_result = {
            "file": os.path.basename(file_path),
            "results": results,
            "confidence_scores": confidence_scores,
            "sources": sources,
            "validation_results": validation_results,
            "csv_validation_results": csv_validation_results
        }
        
        # Cache the result
        await save_to_cache(file_hash, final_result, result_cache)
        
        return final_result
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"file": os.path.basename(file_path), "error": str(e)}

async def process_batch(files, csv_validator=None, batch_size=4):
    """Process files in batches with CSV validation."""
    results = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        logger.info(f"Processing batch of {len(batch)} files")
        batch_results = await asyncio.gather(*(process_file(file, csv_validator) for file in batch))
        results.extend(batch_results)
    return results

@app.post("/extract-invoices/")
async def extract_invoices(data: dict):
    """Extract data from invoice files and validate against CSV database."""
    folder_path = data.get("folder_path")
    csv_path = data.get("csv_path")  # New parameter for CSV file
    
    if not folder_path:
        return {"error": "Folder path must be provided in the request body as 'folder_path'"}

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return {"error": f"Invalid folder path: {folder_path}"}

    # Find all relevant files
    invoice_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    if not invoice_files:
        return {"error": f"No JPG or PNG files found in {folder_path}"}

    logger.info(f"Found {len(invoice_files)} files to process in {folder_path}")

    # Load CSV database if provided
    csv_validator = None
    if csv_path:
        csv_file = Path(csv_path)
        if not csv_file.exists() or not csv_file.is_file():
            return {"error": f"Invalid CSV file path: {csv_path}"}
            
        csv_validator = CSVValidator()
        csv_df = await csv_validator.load_csv_database(csv_path)
        if csv_df is None:
            return {"error": f"Failed to load CSV database from {csv_path}"}
            
        logger.info(f"Loaded CSV database with {len(csv_df)} records for validation")

    # Process in optimal batch sizes
    batch_size = min(4, len(invoice_files))
    start_time = asyncio.get_event_loop().time()
    results = await process_batch([str(file) for file in invoice_files], csv_validator, batch_size)
    end_time = asyncio.get_event_loop().time()
    
    logger.info(f"Processed {len(invoice_files)} files in {end_time - start_time:.2f} seconds")

    # Format results
    formatted_results = []
    for result in results:
        if "error" not in result:
            query_results = {}
            
            # Ensure ALL queries are included
            for query in QUERIES:
                query_text = query["Text"]
                
                # Get values from results, with appropriate fallbacks
                value = result["results"].get(query_text, "Not Found")
                confidence = result["confidence_scores"].get(query_text, 0.0)
                source = result["sources"].get(query_text, "None")
                validated = result["validation_results"].get(query_text, False)
                
                # Format the output correctly
                if value != "Not Found" and confidence > 0:
                    query_results[query_text] = {
                        "value": value,
                        "confidence": confidence,
                        "source": source,
                        "csv_validated": validated
                    }
                else:
                    query_results[query_text] = {
                        "value": "Not Found",
                        "reason": "Lack of information or image quality issue",
                        "source": source,
                        "csv_validated": False
                    }
            
            formatted_results.append({
                "filename": result["file"],
                "data": query_results,
                "csv_validation": {
                    "status": "Validated" if any(r.get("csv_validated", False) for r in query_results.values()) else "Not Found in CSV",
                    "primary_key": result["results"].get("What is the chassis number?", "Not Found")
                }
            })
        else:
            formatted_results.append({
                "filename": result["file"],
                "error": result["error"]
            })

    return {
        "invoices": formatted_results, 
        "processing_time_seconds": end_time - start_time,
        "csv_validation": {
            "status": "Enabled" if csv_validator is not None else "Disabled",
            "records_count": len(csv_validator.csv_df) if csv_validator is not None else 0
        }
    }

# Import validation function
from utils import validate_query_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)