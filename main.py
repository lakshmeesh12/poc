#main.py

from fastapi import FastAPI
import asyncio
import os
from pathlib import Path
import logging
import hashlib
import json
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from utils import get_file_hash, get_cached_result, save_to_cache, optimize_file, validate_query_result, is_pdf_file
from textract import process_with_textract, extract_text_from_textract_response
from llm_text import process_with_llm_text
from llm_image import process_with_llm_media
from validation import EnhancedCSVValidator
from fastapi.middleware.cors import CORSMiddleware
import shutil
from typing import List
from pydantic import BaseModel
import boto3
from fastapi.staticfiles import StaticFiles
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import os
# Load environment variables from .env file
load_dotenv()

# Set up more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

class S3Request(BaseModel):
    bucket: str
    file_keys: List[str]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from your React app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create annotated_images directory if it doesn’t exist
os.makedirs("annotated_images", exist_ok=True)

# Mount the directory for serving static files
annotated_images_dir = os.path.abspath("annotated_images")
app.mount("/annotated", StaticFiles(directory=annotated_images_dir), name="annotated")

# Initialize AWS S3 client
s3_client = boto3.client('s3', region_name="us-west-2")

# Define your custom queries for the adapter
QUERIES = [
    {"Text": "What is the chassis number?"},
    {"Text": "What is the engine number?"},
    {"Text": "What is the make?"},
    {"Text": "What is the model?"},
    {"Text": "What is the customer name?"},
    {"Text": "What is the InsuranceNo or policy number?"},
    {"Text": "What is the Vehicle_Insurance_Company or policy provider company?"}
]

# Result cache
result_cache = {}

# Updated CSV file path for validation
csv_path = "C:/Users/Quadrant/Downloads/Invoice ans Insurance Details.csv"

async def process_file(file_path: str, csv_validator=None):
    """Process a single invoice file with text extraction and validation."""
    try:
        # Check cache first
        file_hash = get_file_hash(file_path)
        cached_result = await get_cached_result(file_hash, result_cache)
        if cached_result:
            logger.info(f"Using cached result for {file_path}")
            return cached_result

        # Initialize results dictionary
        results = {query["Text"]: "Not Found" for query in QUERIES}
        confidence_scores = {query["Text"]: 0.0 for query in QUERIES}
        sources = {query["Text"]: "None" for query in QUERIES}
        validation_results = {query["Text"]: False for query in QUERIES}
        csv_matched_values = {query["Text"]: "Not Found" for query in QUERIES}  # Store CSV values
        
        # Check if file is PDF
        is_pdf = is_pdf_file(file_path)
        logger.info(f"Processing {file_path} - File type: {'PDF' if is_pdf else 'Image'}")
        
        # Optimize file before processing
        optimized_file_bytes = optimize_file(file_path)
        
        # FIRST ATTEMPT: Textract with adapter
        logger.info(f"ATTEMPT 1: Processing {file_path} with Textract adapter")
        
        textract_response, textract_results, textract_confidence, textract_validation = await process_with_textract(file_path, optimized_file_bytes, QUERIES, is_pdf)
        
        raw_extracted_text = extract_text_from_textract_response(textract_response)
        logger.info(f"Extracted {len(raw_extracted_text)} characters of raw text from document")
        
        for query_text, result_text in textract_results.items():
            results[query_text] = result_text
            confidence_scores[query_text] = textract_confidence.get(query_text, 0.0)
            sources[query_text] = "Textract (Attempt 1)"
            validation_results[query_text] = textract_validation.get(query_text, False)
        
        queries_for_second_attempt = []
        for query_text, result_text in results.items():
            if not validation_results.get(query_text, False) or confidence_scores.get(query_text, 0) < 85.0:
                queries_for_second_attempt.append({"Text": query_text})
                logger.info(f"Query '{query_text}' with value '{result_text}' added to second attempt: validation={validation_results.get(query_text, False)}, confidence={confidence_scores.get(query_text, 0)}")
        
        second_attempt_task = None
        if queries_for_second_attempt and raw_extracted_text:
            logger.info(f"ATTEMPT 2: Processing {len(queries_for_second_attempt)} queries with OpenAI using extracted text")
            second_attempt_task = asyncio.create_task(process_with_llm_text(raw_extracted_text, queries_for_second_attempt, 2))
        
        queries_for_third_attempt = []
        for query_text, result_text in results.items():
            if result_text == "Not Found" or not validation_results.get(query_text, False):
                queries_for_third_attempt.append({"Text": query_text})
        
        if second_attempt_task:
            llm_results = await second_attempt_task
            for query_text, result in llm_results.items():
                if isinstance(result, dict):
                    value = result.get("value", "Not Found")
                    if value != "Not Found" and validate_query_result(query_text, value):
                        results[query_text] = value
                        confidence_scores[query_text] = result.get("confidence", 50.0)
                        sources[query_text] = "OpenAI Text (Attempt 2)"
                        validation_results[query_text] = True
                        logger.info(f"Updated with text-based result for '{query_text}': Value='{value}', Valid=True")
        
        queries_for_third_attempt = []
        for query_text, result_text in results.items():
            if result_text == "Not Found" or not validation_results.get(query_text, False):
                queries_for_third_attempt.append({"Text": query_text})
                logger.info(f"Query '{query_text}' with value '{result_text}' added to third attempt: validation={validation_results.get(query_text, False)}")
        
        third_attempt_task = None
        if queries_for_third_attempt:
            logger.info(f"ATTEMPT 3: Processing {len(queries_for_third_attempt)} queries with OpenAI using {'PDF' if is_pdf else 'image'}")
            third_attempt_task = asyncio.create_task(process_with_llm_media(file_path, queries_for_third_attempt, 3))
        
        csv_validation_task = None
        if csv_validator is not None:
            logger.info("Starting parallel CSV validation")
            csv_validation_task = asyncio.create_task(csv_validator.validate_against_csv(results))
        
        if third_attempt_task:
            llm_results = await third_attempt_task
            for query_text, result in llm_results.items():
                if isinstance(result, dict):
                    value = result.get("value", "Not Found")
                    if value != "Not Found" and validate_query_result(query_text, value):
                        results[query_text] = value
                        confidence_scores[query_text] = result.get("confidence", 50.0)
                        sources[query_text] = f"OpenAI {'PDF' if is_pdf else 'Image'} (Attempt 3)"
                        validation_results[query_text] = True
                        logger.info(f"Updated with {'PDF' if is_pdf else 'image'}-based result for '{query_text}': Value='{value}', Valid=True")
                    elif results[query_text] == "Not Found":
                        sources[query_text] = f"OpenAI {'PDF' if is_pdf else 'Image'} (Attempt 3)"
        
        csv_validation_results = {}
        primary_key_used = "None"
        if csv_validation_task:
            try:
                csv_validation_results = await csv_validation_task
                for query_text, validation in csv_validation_results.items():
                    validation_results[query_text] = validation["is_valid"]
                    if validation["is_valid"]:
                        sources[query_text] += " + CSV Validated"
                    csv_matched_values[query_text] = validation["csv_value"]
                    # Store the primary key used (will be the same for all fields in a match)
                    primary_key_used = validation.get("primary_key_used", "None")
                    
                    # Store similarity score if available
                    if "similarity_score" in validation:
                        confidence_scores[query_text] = max(confidence_scores.get(query_text, 0), 
                                                            validation["similarity_score"] * 100)  # Convert to percentage
            except Exception as e:
                logger.error(f"Error during CSV validation: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Include similarity scores in the final result
        final_result = {
            "file": os.path.basename(file_path),
            "file_type": "PDF" if is_pdf else "Image",
            "results": results,
            "confidence_scores": confidence_scores,
            "sources": sources,
            "validation_results": validation_results,
            "csv_validation_results": csv_validation_results,
            "csv_matched_values": csv_matched_values,
            "primary_key_used": primary_key_used,
        }

        # Get mismatched queries (where csv_validated is False)
        mismatched_queries = [query for query, valid in validation_results.items() if not valid]

        # Parse Textract response to get LINE blocks with bounding boxes, grouped by page
        line_blocks = [block for block in textract_response['Blocks'] if block['BlockType'] == 'LINE']
        page_lines = {}
        for block in line_blocks:
            page = block.get('Page', 1)  # Default to page 1 if Page is missing (for images)
            if page not in page_lines:
                page_lines[page] = []
            page_lines[page].append({
                'text': block['Text'],
                'bounding_box': block['Geometry']['BoundingBox']
            })

        # Find bounding boxes for mismatched values
        mismatch_bboxes = {page: [] for page in page_lines}
        for query in mismatched_queries:
            value = results[query]
            if value != "Not Found":
                for page, lines in page_lines.items():
                    for line in lines:
                        # Case-insensitive search for the value in the line text
                        if value.lower() in line['text'].lower():
                            mismatch_bboxes[page].append(line['bounding_box'])

        # Generate annotated images
        annotated_images = []
        if is_pdf:
            # Convert PDF to a list of PIL images
            pdf_images = convert_from_path(file_path)
            for page_num, image in enumerate(pdf_images, start=1):
                if page_num in mismatch_bboxes and mismatch_bboxes[page_num]:
                    draw = ImageDraw.Draw(image)
                    width, height = image.size
                    for bbox in mismatch_bboxes[page_num]:
                        # Convert Textract bounding box coordinates to pixel values
                        left = bbox['Left'] * width
                        top = bbox['Top'] * height
                        right = (bbox['Left'] + bbox['Width']) * width
                        bottom = (bbox['Top'] + bbox['Height']) * height
                        draw.rectangle([left, top, right, bottom], outline="red", width=3)
                    # Save annotated image with a unique name
                    annotated_path = f"annotated_images/{file_hash}_page{page_num}.png"
                    image.save(annotated_path)
                    annotated_images.append(f"/annotated/{file_hash}_page{page_num}.png")
        else:
            # Process single image
            image = Image.open(file_path)
            if 1 in mismatch_bboxes and mismatch_bboxes[1]:
                draw = ImageDraw.Draw(image)
                width, height = image.size
                for bbox in mismatch_bboxes[1]:
                    left = bbox['Left'] * width
                    top = bbox['Top'] * height
                    right = (bbox['Left'] + bbox['Width']) * width
                    bottom = (bbox['Top'] + bbox['Height']) * height
                    draw.rectangle([left, top, right, bottom], outline="red", width=3)
            annotated_path = f"annotated_images/{file_hash}.png"
            image.save(annotated_path)
            annotated_images.append(f"/annotated/{file_hash}.png")

        # Add annotated image paths to final_result
        final_result["annotated_images"] = annotated_images
        
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

def convert_numpy_types(obj):
    """Convert numpy types to Python standard types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(i) for i in obj)
    else:
        return obj

@app.post("/extract-invoices/", response_model=None)
async def extract_invoices(files: List[UploadFile] = File(...)):
    """Extract data from uploaded invoice files and validate against hardcoded CSV database."""
    # Temporary directory to store uploaded files
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)

    # Save uploaded files locally
    file_paths = []
    try:
        for file in files:
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(str(file_path))
            logger.info(f"Saved uploaded file to {file_path}")

        if not file_paths:
            return {"error": "No files uploaded"}

        # Load hardcoded CSV database
        csv_validator = None
        csv_file = Path(csv_path)
        if csv_file.exists() and csv_file.is_file():
            # Use the enhanced validator instead of the basic one
            csv_validator = EnhancedCSVValidator()
            csv_df = await csv_validator.load_csv_database(csv_path)
            if csv_df is None:
                logger.error(f"Failed to load hardcoded CSV database from {csv_path}")
            else:
                logger.info(f"Loaded hardcoded CSV database with {len(csv_df)} records for validation")
        else:
            logger.warning(f"Hardcoded CSV file not found at {csv_path}, proceeding without validation")

        # Process files
        batch_size = min(4, len(file_paths))
        start_time = asyncio.get_event_loop().time()
        results = await process_batch(file_paths, csv_validator, batch_size)
        end_time = asyncio.get_event_loop().time()
        
        logger.info(f"Processed {len(file_paths)} files in {end_time - start_time:.2f} seconds")

        # Format results
        formatted_results = []
        for result in results:
            if "error" not in result:
                query_results = {}
                for query in QUERIES:
                    query_text = query["Text"]
                    value = result["results"].get(query_text, "Not Found")
                    confidence = result["confidence_scores"].get(query_text, 0.0)
                    source = result["sources"].get(query_text, "None")
                    validated = result["validation_results"].get(query_text, False)
                    csv_value = result["csv_matched_values"].get(query_text, "Not Found")
                    
                    if value != "Not Found" and confidence > 0:
                        query_results[query_text] = {
                            "value": value,
                            "confidence": confidence,
                            "source": source,
                            "csv_validated": validated,
                            "csv_value": csv_value
                        }
                    else:
                        query_results[query_text] = {
                            "value": "Not Found",
                            "reason": "Lack of information or quality issue",
                            "source": source,
                            "csv_validated": False,
                            "csv_value": csv_value
                        }
                
                formatted_results.append({
                    "filename": result["file"],
                    "file_type": result.get("file_type", "Image"),
                    "data": query_results,
                    "csv_validation": {
                        "status": "Validated" if any(r.get("csv_validated", False) for r in query_results.values()) else "Not Found in CSV",
                        "primary_key": query_results.get(result.get("primary_key_used", "None"), {}).get("value", "Not Found")
                    },
                    "annotated_images": result.get("annotated_images", [])  # Add this line
                })
            else:
                formatted_results.append({
                    "filename": result["file"],
                    "error": result["error"]
                })

        final_response = {
            "invoices": formatted_results,
            "processing_time_seconds": end_time - start_time,
            "file_types_processed": {
                "images": len([r for r in formatted_results if "file_type" in r and r["file_type"] == "Image"]),
                "pdfs": len([r for r in formatted_results if "file_type" in r and r["file_type"] == "PDF"])
            },
            "csv_validation": {
                "status": "Enabled" if csv_validator is not None else "Disabled",
                "records_count": len(csv_validator.csv_df) if csv_validator is not None else 0
            }
        }
        final_response = convert_numpy_types(final_response)
        return final_response

    finally:
        # Clean up temporary files
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            logger.info(f"Removed temporary directory: {temp_dir}")

@app.get("/list-buckets/")
async def list_buckets():
    """List all available S3 buckets for the user to select."""
    try:
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        return {"buckets": buckets}
    except Exception as e:
        logger.error(f"Error listing buckets: {str(e)}")
        return {"error": str(e)}

@app.get("/list-files/")
async def list_files(bucket: str):
    """List all files in the specified S3 bucket."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket)
        if 'Contents' in response:
            files = [obj['Key'] for obj in response['Contents']]
            return {"files": files}
        else:
            return {"files": []}
    except Exception as e:
        logger.error(f"Error listing files in bucket {bucket}: {str(e)}")
        return {"error": str(e)}

@app.post("/extract-invoices-s3/")
async def extract_invoices_s3(request: S3Request):
    bucket = request.bucket
    file_keys = request.file_keys
    
    """Process selected files from an S3 bucket."""
    temp_dir = Path("temp_s3_downloads")
    temp_dir.mkdir(exist_ok=True)
    file_paths = []

    try:
        # Download files from S3
        for key in file_keys:
            file_path = temp_dir / key.replace('/', '_')  # Flatten file structure
            s3_client.download_file(bucket, key, str(file_path))
            file_paths.append(str(file_path))
            logger.info(f"Downloaded {key} from S3 to {file_path}")

        # Load CSV validator if available
        csv_validator = None
        csv_file = Path(csv_path)
        if csv_file.exists() and csv_file.is_file():
            csv_validator = CSVValidator()
            csv_df = await csv_validator.load_csv_database(csv_path)
            if csv_df is None:
                logger.error(f"Failed to load CSV database from {csv_path}")
            else:
                logger.info(f"Loaded CSV database with {len(csv_df)} records")

        # Process files using existing logic
        batch_size = min(4, len(file_paths))
        start_time = asyncio.get_event_loop().time()
        results = await process_batch(file_paths, csv_validator, batch_size)
        end_time = asyncio.get_event_loop().time()

        logger.info(f"Processed {len(file_paths)} S3 files in {end_time - start_time:.2f} seconds")

        # Format results (same as local upload)
        formatted_results = []
        for result in results:
            if "error" not in result:
                query_results = {}
                for query in QUERIES:
                    query_text = query["Text"]
                    value = result["results"].get(query_text, "Not Found")
                    confidence = result["confidence_scores"].get(query_text, 0.0)
                    source = result["sources"].get(query_text, "None")
                    validated = result["validation_results"].get(query_text, False)
                    csv_value = result["csv_matched_values"].get(query_text, "Not Found")

                    if value != "Not Found" and confidence > 0:
                        query_results[query_text] = {
                            "value": value,
                            "confidence": confidence,
                            "source": source,
                            "csv_validated": validated,
                            "csv_value": csv_value
                        }
                    else:
                        query_results[query_text] = {
                            "value": "Not Found",
                            "reason": "Lack of information or quality issue",
                            "source": source,
                            "csv_validated": False,
                            "csv_value": csv_value
                        }

                formatted_results.append({
                    "filename": result["file"],
                    "file_type": result.get("file_type", "Image"),
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

        final_response = {
            "invoices": formatted_results,
            "processing_time_seconds": end_time - start_time,
            "file_types_processed": {
                "images": len([r for r in formatted_results if "file_type" in r and r["file_type"] == "Image"]),
                "pdfs": len([r for r in formatted_results if "file_type" in r and r["file_type"] == "PDF"])
            },
            "csv_validation": {
                "status": "Enabled" if csv_validator is not None else "Disabled",
                "records_count": len(csv_validator.csv_df) if csv_validator is not None else 0
            }
        }
        final_response = convert_numpy_types(final_response)
        return final_response

    finally:
        # Clean up temporary files
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary S3 file: {file_path}")
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
            logger.info(f"Removed temporary S3 directory: {temp_dir}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)