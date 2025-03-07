import os
import json
import asyncio
import logging
import base64
from openai import OpenAI
from utils import optimize_image, get_pattern_description
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Configure OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def process_with_llm_image(file_path: str, queries_to_process: list, attempt: int):
    """Process specified queries with GPT-4o using image."""
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
    except Exception as e:
        logger.error(f"Error processing with LLM image (Attempt {attempt}): {str(e)}")
        return {query["Text"]: {"value": "Not Found", "reason": f"LLM processing error: {str(e)}"} for query in queries_to_process}