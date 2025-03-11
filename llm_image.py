import os
import json
import asyncio
import logging
import base64
from openai import OpenAI
from utils import optimize_image, optimize_file, is_pdf_file, get_pattern_description
from dotenv import load_dotenv
import tempfile
import fitz  # PyMuPDF for PDF handling

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Configure OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def process_with_llm_media(file_path: str, queries_to_process: list, attempt: int):
    """
    Process specified queries with GPT-4o using media (image or PDF).
    Supports both image files and PDF documents.
    """
    try:
        # Check if file is PDF
        is_pdf = is_pdf_file(file_path)
        logger.info(f"Processing {file_path} with LLM media (Attempt {attempt}) - File type: {'PDF' if is_pdf else 'Image'}")
        
        # For PDFs, we need to handle it differently
        if is_pdf:
            return await process_pdf_with_llm(file_path, queries_to_process, attempt)
        else:
            # Handle image files
            return await process_image_with_llm(file_path, queries_to_process, attempt)
    except Exception as e:
        logger.error(f"Error processing with LLM media (Attempt {attempt}): {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {query["Text"]: {"value": "Not Found", "reason": f"LLM processing error: {str(e)}"} for query in queries_to_process}

async def process_image_with_llm(file_path: str, queries_to_process: list, attempt: int):
    """Process image file with GPT-4o."""
    try:
        # Use optimized image
        image_bytes = optimize_image(file_path)
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # More concise prompt
        prompt = (
            f"Extract the following information from the provided invoice image and return as JSON with 'value' and 'confidence' (0-100) for each query. "
            f"If information can't be extracted, include 'reason' instead of 'confidence'. "
            f"Return only JSON, no markdown. Queries:\n"
        )
        for query in queries_to_process:
            prompt += f"- {query['Text']}\n"
            pattern_description = get_pattern_description(query['Text'])
            if pattern_description:
                prompt += f"  (FORMAT REQUIREMENT: {pattern_description})\n"
        
        prompt += (
            "Example format:\n"
            "{\n"
            "  \"What is the chassis number?\": {\"value\": \"ABC123XYZ456789\", \"confidence\": 95.0},\n"
            "  \"What is the color?\": {\"value\": \"Not Found\", \"reason\": \"No color information in image\"}\n"
            "}"
        )

        # Use a task to run this potentially slow operation asynchronously
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ]
                    }
                ],
                max_tokens=500
            )
        )

        llm_response = response.choices[0].message.content.strip()
        logger.info(f"LLM image-based response received (Attempt {attempt})")

        # Clean up and parse the response
        return parse_llm_response(llm_response, queries_to_process)
    
    except Exception as e:
        logger.error(f"Error processing image with LLM (Attempt {attempt}): {str(e)}")
        return {query["Text"]: {"value": "Not Found", "reason": f"LLM image processing error: {str(e)}"} for query in queries_to_process}

async def process_pdf_with_llm(file_path: str, queries_to_process: list, attempt: int):
    """Process PDF file with GPT-4o."""
    try:
        # Get optimized PDF file bytes
        pdf_bytes = optimize_file(file_path)
        
        # For PDFs, we have two approaches:
        # 1. Try to render the first page as an image and send that
        # 2. If that fails, send the PDF directly (if supported by the API)
        
        try:
            # First approach: Convert first page to image
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_bytes)
                temp_file_path = temp_file.name
            
            # Use PyMuPDF to render first page as image
            doc = fitz.open(temp_file_path)
            first_page = doc.load_page(0)  # Load first page
            pix = first_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
            
            # Get image bytes from pixmap
            image_bytes = pix.tobytes("jpeg")
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Clean up
            doc.close()
            os.unlink(temp_file_path)
            
            # More concise prompt for PDF
            prompt = (
                f"Extract the following information from the provided PDF document (first page shown as image) and return as JSON with 'value' and 'confidence' (0-100) for each query. "
                f"If information can't be extracted, include 'reason' instead of 'confidence'. "
                f"Return only JSON, no markdown. Queries:\n"
            )
            for query in queries_to_process:
                prompt += f"- {query['Text']}\n"
                pattern_description = get_pattern_description(query['Text'])
                if pattern_description:
                    prompt += f"  (FORMAT REQUIREMENT: {pattern_description})\n"
            
            prompt += (
                "Example format:\n"
                "{\n"
                "  \"What is the chassis number?\": {\"value\": \"ABC123XYZ456789\", \"confidence\": 95.0},\n"
                "  \"What is the color?\": {\"value\": \"Not Found\", \"reason\": \"No color information in PDF\"}\n"
                "}"
            )

            # Use a task to run this potentially slow operation asynchronously
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                            ]
                        }
                    ],
                    max_tokens=500
                )
            )

            llm_response = response.choices[0].message.content.strip()
            logger.info(f"LLM PDF-based (image method) response received (Attempt {attempt})")

            # Clean up and parse the response
            return parse_llm_response(llm_response, queries_to_process)
            
        except Exception as pdf_img_error:
            # If converting to image fails, try to send the PDF directly if supported
            logger.warning(f"Failed to process PDF as image: {str(pdf_img_error)}. Falling back to base64 PDF method")
            
            # Encode the PDF for direct sending
            encoded_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            
            prompt = (
                f"Extract the following information from the provided PDF document and return as JSON with 'value' and 'confidence' (0-100) for each query. "
                f"If information can't be extracted, include 'reason' instead of 'confidence'. "
                f"Return only JSON, no markdown. Queries:\n"
            )
            for query in queries_to_process:
                prompt += f"- {query['Text']}\n"
                pattern_description = get_pattern_description(query['Text'])
                if pattern_description:
                    prompt += f"  (FORMAT REQUIREMENT: {pattern_description})\n"
            
            prompt += (
                "Example format:\n"
                "{\n"
                "  \"What is the chassis number?\": {\"value\": \"ABC123XYZ456789\", \"confidence\": 95.0},\n"
                "  \"What is the color?\": {\"value\": \"Not Found\", \"reason\": \"No color information in PDF\"}\n"
                "}"
            )

            # Try with direct PDF upload - may not work if API doesn't support PDF
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{encoded_pdf}"}}
                                ]
                            }
                        ],
                        max_tokens=500
                    )
                )

                llm_response = response.choices[0].message.content.strip()
                logger.info(f"LLM PDF-based (direct PDF method) response received (Attempt {attempt})")

                # Clean up and parse the response
                return parse_llm_response(llm_response, queries_to_process)
                
            except Exception as pdf_direct_error:
                # If both methods fail, log the error and return not found
                logger.error(f"Both PDF processing methods failed: Image error: {str(pdf_img_error)}, Direct PDF error: {str(pdf_direct_error)}")
                return {query["Text"]: {"value": "Not Found", "reason": f"PDF processing failed with both methods"} for query in queries_to_process}
    
    except Exception as e:
        logger.error(f"Error processing PDF with LLM (Attempt {attempt}): {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {query["Text"]: {"value": "Not Found", "reason": f"LLM PDF processing error: {str(e)}"} for query in queries_to_process}

def parse_llm_response(llm_response, queries_to_process):
    """Helper function to parse and clean LLM response."""
    # Clean up the response
    cleaned_response = llm_response
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()

    try:
        results = json.loads(cleaned_response)
        return results
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
        return {query["Text"]: {"value": "Not Found", "reason": f"LLM response parsing error: {str(e)}"} for query in queries_to_process}