# tamper_detection.py

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
import boto3
import logging
import base64
import hashlib
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from pdf2image import convert_from_path
import tempfile
import os

logger = logging.getLogger(__name__)

class TamperDetector:
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize the tamper detector with configurable threshold."""
        self.confidence_threshold = confidence_threshold
        self.rekognition_client = boto3.client('rekognition')
        
    async def detect_tampering(self, file_path: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        tampering_detected = False
        confidence_score = 0.0
        tampering_methods = []
        
        is_pdf = extracted_data.get("is_pdf", False)
        image_paths = []
        temp_dir = None
        try:
            # Handle PDF conversion to images
            if is_pdf:
                temp_dir = tempfile.mkdtemp()
                images = convert_from_path(file_path)
                for i, img in enumerate(images):
                    img_path = os.path.join(temp_dir, f"page_{i}.jpg")
                    img.save(img_path, "JPEG")
                    image_paths.append(img_path)
                logger.info(f"Converted PDF {file_path} to {len(image_paths)} image(s)")
            else:
                image_paths = [file_path]
            
            # Process only the first page/image for simplicity
            img_path = image_paths[0]
            method_results = []
            for method in [
                self._check_ela_anomalies,
                self._check_inconsistent_fonts,
                self._check_text_alignment,
                self._check_metadata_consistency,
                self._check_data_consistency
            ]:
                result = await method(img_path, extracted_data)
                method_results.append(result)
                if result["tampering_detected"]:
                    tampering_detected = True
                    tampering_methods.append(result["method"])
                    confidence_score = max(confidence_score, result["confidence"])
            
            # Annotate the image with bounding boxes
            method_colors = {
                "font_consistency_check": (0, 0, 255),  # Red
                "text_alignment_check": (255, 0, 0),    # Blue
                "error_level_analysis": (0, 255, 0)     # Green
                # Metadata and data consistency don’t have boxes
            }
            
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Failed to load image at {img_path}")
                annotated_image_base64 = ""
            else:
                height, width = img.shape[:2]
                all_bboxes = []
                for result in method_results:
                    if result["tampering_detected"] and "bounding_boxes" in result and result["bounding_boxes"]:
                        for bbox in result["bounding_boxes"]:
                            x = int(bbox['Left'] * width)
                            y = int(bbox['Top'] * height)
                            w = int(bbox['Width'] * width)
                            h = int(bbox['Height'] * height)
                            all_bboxes.append((x, y, x + w, y + h, result["method"]))
                
                # Draw rectangles and labels on the image
                for x1, y1, x2, y2, method in all_bboxes:
                    color = method_colors.get(method, (0, 0, 0))  # Default black
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, method.split('_')[0], (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Encode the annotated image as base64
                _, buffer = cv2.imencode('.jpg', img)
                annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Compile results
            results = {
                "tampering_detected": tampering_detected,
                "confidence_score": confidence_score,
                "tampering_methods": tampering_methods,
                "method_details": method_results,
                "annotated_image": annotated_image_base64  # Include the annotated image
            }
        
        finally:
            # Clean up temporary files
            if temp_dir and os.path.exists(temp_dir):
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                os.rmdir(temp_dir)
                logger.info(f"Cleaned up temporary directory {temp_dir}")
        
        return results
    
    async def _check_ela_anomalies(self, file_path: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            ela_image = self._perform_ela(file_path)
            ela_array = np.array(ela_image)
            if len(ela_array.shape) == 3:
                ela_gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
            else:
                ela_gray = ela_array
            
            _, thresh = cv2.threshold(ela_gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bounding_boxes = []
            img = Image.open(file_path)
            width, height = img.size
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox = {
                        'Left': x / width,
                        'Top': y / height,
                        'Width': w / width,
                        'Height': h / height
                    }
                    bounding_boxes.append(bbox)
            
            tampering_score = self._analyze_ela(ela_image)
            tampering_detected = tampering_score > self.confidence_threshold
            
            return {
                "method": "error_level_analysis",
                "tampering_detected": tampering_detected,
                "confidence": tampering_score,
                "details": f"ELA detected editing with confidence {tampering_score:.2f}",
                "bounding_boxes": bounding_boxes  # Added bounding boxes
            }
        except Exception as e:
            logger.error(f"Error in ELA analysis: {str(e)}")
            return {
                "method": "error_level_analysis",
                "tampering_detected": False,
                "confidence": 0.0,
                "details": f"ELA analysis failed: {str(e)}",
                "bounding_boxes": []
            }
        
    def _perform_ela(self, image_path: str, quality: int = 90) -> Image.Image:
        """Perform Error Level Analysis on an image."""
        original = Image.open(image_path).convert('RGB')
        temp_buffer = io.BytesIO()
        original.save(temp_buffer, format="JPEG", quality=quality)
        temp_buffer.seek(0)
        compressed = Image.open(temp_buffer).convert('RGB')
        ela_image = ImageChops.difference(original, compressed)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff > 0:
            scale = 255.0 / max_diff
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        return ela_image
    
    def _analyze_ela(self, ela_image: Image.Image) -> float:
        """Analyze ELA image for evidence of tampering."""
        ela_array = np.array(ela_image)
        if len(ela_array.shape) == 3:
            ela_gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
        else:
            ela_gray = ela_array
        mean_intensity = np.mean(ela_gray)
        std_intensity = np.std(ela_gray)
        _, thresh = cv2.threshold(ela_gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contour_count = sum(1 for c in contours if cv2.contourArea(c) > 100)
        confidence_score = min(1.0, (
            0.4 * (mean_intensity / 50) +
            0.3 * (std_intensity / 40) +
            0.3 * (large_contour_count / 5)
        ))
        return confidence_score
    
    async def _check_inconsistent_fonts(self, file_path: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with open(file_path, 'rb') as image_file:
                image_bytes = image_file.read()
            response = self.rekognition_client.detect_text(Image={'Bytes': image_bytes})
            text_detections = response.get('TextDetections', [])
            
            font_groups = {}
            for detection in text_detections:
                if detection['Type'] == 'WORD' and detection['Confidence'] > 80:
                    bbox = detection['Geometry']['BoundingBox']
                    height = bbox['Height']
                    font_sig = f"{round(height * 1000)}"
                    if font_sig not in font_groups:
                        font_groups[font_sig] = []
                    font_groups[font_sig].append(detection)
            
            anomalies = []
            for font_sig, detections in font_groups.items():
                if 1 <= len(detections) <= 3 and len(font_groups) > 2:
                    for det in detections:
                        if any(char.isdigit() for char in det['DetectedText']):
                            anomalies.append(det['Geometry']['BoundingBox'])
            
            tampering_detected = len(anomalies) > 0
            confidence = min(1.0, len(anomalies) * 0.3)
            
            return {
                "method": "font_consistency_check",
                "tampering_detected": tampering_detected,
                "confidence": confidence,
                "details": f"Found {len(anomalies)} text regions with inconsistent fonts" if anomalies else "No font inconsistencies detected",
                "bounding_boxes": anomalies  # Added bounding boxes
            }
        except Exception as e:
            logger.error(f"Error in font consistency check: {str(e)}")
            return {
                "method": "font_consistency_check",
                "tampering_detected": False,
                "confidence": 0.0,
                "details": f"Font analysis failed: {str(e)}",
                "bounding_boxes": []
            }
    
    async def _check_text_alignment(self, file_path: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with open(file_path, 'rb') as image_file:
                image_bytes = image_file.read()
            response = self.rekognition_client.detect_text(Image={'Bytes': image_bytes})
            text_detections = response.get('TextDetections', [])
            
            line_detections = [det for det in text_detections if det['Type'] == 'LINE']
            if not line_detections:
                return {
                    "method": "text_alignment_check",
                    "tampering_detected": False,
                    "confidence": 0.0,
                    "details": "No text lines detected",
                    "bounding_boxes": []
                }
            
            vertical_positions = [det['Geometry']['BoundingBox']['Top'] for det in line_detections]
            vertical_positions.sort()
            spacings = [vertical_positions[i+1] - vertical_positions[i] for i in range(len(vertical_positions)-1)]
            median_spacing = np.median(spacings)
            
            irregularities = []
            for i in range(len(spacings)):
                if abs(spacings[i] - median_spacing) > 0.5 * median_spacing:
                    irregularities.append(line_detections[i]['Geometry']['BoundingBox'])
                    irregularities.append(line_detections[i+1]['Geometry']['BoundingBox'])
            
            tampering_detected = len(irregularities) > 0
            confidence = min(1.0, len(irregularities) * 0.1)
            
            return {
                "method": "text_alignment_check",
                "tampering_detected": tampering_detected,
                "confidence": confidence,
                "details": f"Found {len(irregularities)//2} text alignment irregularities",
                "bounding_boxes": irregularities  # Added bounding boxes
            }
        except Exception as e:
            logger.error(f"Error in text alignment check: {str(e)}")
            return {
                "method": "text_alignment_check",
                "tampering_detected": False,
                "confidence": 0.0,
                "details": f"Text alignment analysis failed: {str(e)}",
                "bounding_boxes": []
            }
    
    async def _check_metadata_consistency(self, file_path: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image = Image.open(file_path)
            exif_data = image._getexif() if hasattr(image, '_getexif') else None
            
            if exif_data is None or len(exif_data) < 3:
                return {
                    "method": "metadata_check",
                    "tampering_detected": True,
                    "confidence": 0.6,
                    "details": "Missing or limited metadata, possibly stripped during editing",
                    "bounding_boxes": []  # No spatial data
                }
            
            return {
                "method": "metadata_check",
                "tampering_detected": False,
                "confidence": 0.0,
                "details": "Metadata appears consistent",
                "bounding_boxes": []  # No spatial data
            }
        except Exception as e:
            logger.error(f"Error in metadata check: {str(e)}")
            return {
                "method": "metadata_check",
                "tampering_detected": False,
                "confidence": 0.0,
                "details": f"Metadata analysis failed: {str(e)}",
                "bounding_boxes": []
            }
    
    async def _check_data_consistency(self, file_path: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for logical inconsistencies in the extracted data."""
        try:
            inconsistencies = []
            model_name = extracted_data.get("What is the model?", "")
            chassis_num = extracted_data.get("What is the chassis number?", "")
            engine_num = extracted_data.get("What is the engine number?", "")
            
            if chassis_num and engine_num:
                if chassis_num.startswith("ME3") and not engine_num.startswith("J"):
                    inconsistencies.append("Engine number format inconsistent with chassis number")
            
            tampering_detected = len(inconsistencies) > 0
            confidence = min(1.0, len(inconsistencies) * 0.5)
            
            return {
                "method": "data_consistency_check",
                "tampering_detected": tampering_detected,
                "confidence": confidence,
                "details": "; ".join(inconsistencies) if inconsistencies else "No data inconsistencies detected"
            }
        except Exception as e:
            logger.error(f"Error in data consistency check: {str(e)}")
            return {
                "method": "data_consistency_check",
                "tampering_detected": False,
                "confidence": 0.0,
                "details": f"Data consistency analysis failed: {str(e)}"
            }
        
    