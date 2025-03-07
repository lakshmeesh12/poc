import boto3
import asyncio
import logging
from utils import validate_query_result

# Set up logging
logger = logging.getLogger(__name__)

# Configure Textract client for us-west-2 region
textract = boto3.client('textract', region_name="us-west-2")
adapter_arn = "arn:aws:textract:us-west-2:296062547225:adapter/invoice-fraud-adapter/c13faff24c83"

def extract_text_from_textract_response(response):
    """Extract all the raw text from a Textract response."""
    text_blocks = []
    for block in response.get("Blocks", []):
        if block.get("BlockType") == "LINE" and "Text" in block:
            text_blocks.append(block["Text"])
    return "\n".join(text_blocks)

async def process_with_textract(file_path, optimized_image_bytes, queries):
    """Process file with Textract adapter and return results."""
    try:
        # Process with Textract asynchronously
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: textract.analyze_document(
                Document={'Bytes': optimized_image_bytes},
                FeatureTypes=['QUERIES', 'TABLES', 'FORMS'],  # Include more features to get all text
                QueriesConfig={"Queries": queries},
                AdaptersConfig={"Adapters": [{"AdapterId": adapter_arn.split('/')[-1], "Version": "1"}]}
            )
        )

        logger.info(f"Textract response received for {file_path}")
        
        # Initialize result containers
        results = {query["Text"]: "Not Found" for query in queries}
        confidence_scores = {query["Text"]: 0.0 for query in queries}
        validation_results = {query["Text"]: False for query in queries}
        
        # Build lookup maps for faster processing
        query_id_to_text = {
            block["Id"]: block["Query"]["Text"] 
            for block in response["Blocks"] 
            if block["BlockType"] == "QUERY" and "Query" in block
        }
        
        # Map of result blocks to their query IDs
        query_result_map = {}
        for block in response["Blocks"]:
            if block["BlockType"] == "QUERY" and "Relationships" in block:
                for relationship in block["Relationships"]:
                    if relationship["Type"] == "ANSWER":
                        for result_id in relationship["Ids"]:
                            query_result_map[result_id] = block["Id"]
        
        # Process results more efficiently
        for block in response["Blocks"]:
            if block["BlockType"] == "QUERY_RESULT":
                query_id = query_result_map.get(block["Id"])
                if query_id and query_id in query_id_to_text:
                    query_text = query_id_to_text[query_id]
                    result_text = block.get("Text", "Not Found")
                    result_confidence = block.get("Confidence", 0.0)
                    
                    results[query_text] = result_text
                    confidence_scores[query_text] = result_confidence
                    
                    # Run validation
                    validation_pass = validate_query_result(query_text, result_text)
                    validation_results[query_text] = validation_pass
        
        return response, results, confidence_scores, validation_results
        
    except Exception as e:
        logger.error(f"Error in Textract processing for {file_path}: {str(e)}")
        # Return empty results in case of error
        results = {query["Text"]: "Not Found" for query in queries}
        confidence_scores = {query["Text"]: 0.0 for query in queries}
        validation_results = {query["Text"]: False for query in queries}
        return {}, results, confidence_scores, validation_results