import logging
import os
import aiohttp
import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import asyncio
# Set up logging
logger = logging.getLogger(__name__)

class OcrolusDocumentValidator:
    """Class to handle document tampering detection using Ocrolus API."""
    
    def __init__(self, client_id: str, client_secret: str):
        """Initialize with Ocrolus API credentials."""
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_token = None
        self.token_expires_at = 0
        self.base_url = "https://api.ocrolus.com/v1"
    
    async def _get_auth_token(self) -> str:
        """Get or refresh authentication token for Ocrolus API."""
        now = int(__import__('time').time())
        
        # Return existing token if still valid
        if self.auth_token and now < self.token_expires_at:
            return self.auth_token
            
        # Get new token
        logger.info("Getting new Ocrolus authentication token")
        auth_url = f"{self.base_url}/auth/token"
        
        async with aiohttp.ClientSession() as session:
            auth_data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials"
            }
            
            try:
                async with session.post(auth_url, json=auth_data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to get Ocrolus auth token. Status: {response.status}, Response: {error_text}")
                        raise Exception(f"Failed to authenticate with Ocrolus API: {error_text}")
                    
                    auth_response = await response.json()
                    self.auth_token = auth_response.get("access_token")
                    # Token typically valid for 1 hour, set expiry to be a bit earlier to be safe
                    expires_in = auth_response.get("expires_in", 3600)
                    self.token_expires_at = now + expires_in - 60
                    
                    return self.auth_token
            except Exception as e:
                logger.error(f"Error during Ocrolus authentication: {str(e)}")
                raise
    
    async def _encode_file(self, file_path: str) -> str:
        """Convert file to base64 for API transmission."""
        try:
            with open(file_path, "rb") as file:
                file_content = file.read()
                encoded = base64.b64encode(file_content).decode("utf-8")
                return encoded
        except Exception as e:
            logger.error(f"Error encoding file {file_path}: {str(e)}")
            raise
    
    async def detect_tampering(self, file_path: str) -> Dict:
        """
        Detect if a document has been tampered with using Ocrolus API.
        
        Args:
            file_path: Path to the image file to check
            
        Returns:
            Dictionary with tampering detection results
        """
        try:
            # Get authentication token
            token = await self._get_auth_token()
            
            # Prepare file data
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1].lower()
            
            # Encode file
            encoded_file = await self._encode_file(file_path)
            
            # Determine content type based on file extension
            content_type = "image/jpeg"
            if file_extension == ".png":
                content_type = "image/png"
            
            # Build request payload
            payload = {
                "document": {
                    "name": file_name,
                    "content_type": content_type,
                    "content": encoded_file
                },
                "analysis_types": ["tampering"]
            }
            
            # Send request
            url = f"{self.base_url}/documents/analyze"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                logger.info(f"Sending tampering detection request for {file_name}")
                
                async with session.post(url, headers=headers, json=payload) as response:
                    response_text = await response.text()
                    
                    if response.status != 200:
                        logger.error(f"Ocrolus API error: {response.status}, Response: {response_text}")
                        return {
                            "success": False,
                            "file": file_name,
                            "error": f"API error: {response.status}",
                            "details": response_text
                        }
                    
                    result = json.loads(response_text)
                    
                    # Process the tampering analysis results
                    tampering_results = self._process_tampering_results(result, file_name)
                    return tampering_results
        
        except Exception as e:
            logger.error(f"Error in tampering detection for {file_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "file": os.path.basename(file_path),
                "error": str(e)
            }
    
    def _process_tampering_results(self, api_response: Dict, file_name: str) -> Dict:
        """Process and format the tampering detection results from Ocrolus API."""
        try:
            # Extract tampering analysis from response
            tampering_analysis = None
            if "analyses" in api_response:
                for analysis in api_response.get("analyses", []):
                    if analysis.get("type") == "tampering":
                        tampering_analysis = analysis
                        break
            
            if not tampering_analysis:
                return {
                    "success": True,
                    "file": file_name,
                    "tampering_detected": False,
                    "confidence": 0,
                    "details": "No tampering analysis found in API response"
                }
            
            # Extract tampering details
            tampering_detected = tampering_analysis.get("results", {}).get("tampering_detected", False)
            confidence = tampering_analysis.get("results", {}).get("confidence", 0)
            tampering_type = tampering_analysis.get("results", {}).get("tampering_type", "None")
            
            # List specific issues found
            issues = []
            for issue in tampering_analysis.get("results", {}).get("issues", []):
                issues.append({
                    "type": issue.get("type", "Unknown"),
                    "confidence": issue.get("confidence", 0),
                    "description": issue.get("description", "No description")
                })
            
            return {
                "success": True,
                "file": file_name,
                "tampering_detected": tampering_detected,
                "confidence": confidence,
                "tampering_type": tampering_type,
                "issues": issues
            }
        
        except Exception as e:
            logger.error(f"Error processing tampering results: {str(e)}")
            return {
                "success": False,
                "file": file_name,
                "error": f"Failed to process tampering results: {str(e)}"
            }

    async def batch_detect_tampering(self, file_paths: List[str], batch_size: int = 4) -> List[Dict]:
        """Process multiple files for tampering detection in batches."""
        results = []
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i+batch_size]
            logger.info(f"Processing tampering detection batch of {len(batch)} files")
            
            # Process files in parallel
            async with aiohttp.ClientSession() as session:
                tasks = [self.detect_tampering(file) for file in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions and format results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error in tampering detection: {str(result)}")
                        results.append({
                            "success": False,
                            "file": os.path.basename(batch[j]),
                            "error": str(result)
                        })
                    else:
                        results.append(result)
        
        return results