import hashlib
import re
import logging
from PIL import Image
import io

# Set up logging
logger = logging.getLogger(__name__)

# Validation patterns for each query
VALIDATION_PATTERNS = {
    "What is the chassis number?": r"^[A-Z0-9]{10,17}$",  # Alphanumeric, uppercase, 10-17 chars (e.g., ME3J3D5FCR1013400)
    "What is the engine number?": r"^[A-Z0-9]{8,15}$",     # Alphanumeric, uppercase, 8-15 chars, no spaces (e.g., J3A5FCR1230192)
    "What is the make?": r"^[A-Za-z\s]+$",                # Uppercase or lowercase letters and spaces (e.g., Pulser 130)
    "What is the model?": r"^[A-Za-z0-9\s-]+$",          # Alphanumeric (upper/lowercase), spaces, and hyphens (e.g., Pulser 130)
    "What is the color?": r"^[A-Za-z\s]+$",              # Uppercase or lowercase letters and spaces (e.g., DEEP RED)
    "What is the customer name?": r"^[A-Za-z\s\.\']+$"   # Uppercase or lowercase letters, spaces, dots, and apostrophes (e.g., SURRAMAN)
}

def get_file_hash(file_path: str) -> str:
    """Generate hash for a file to use as cache key."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

async def get_cached_result(file_hash: str, cache_dict: dict):
    """Get cached result if available."""
    return cache_dict.get(file_hash)

async def save_to_cache(file_hash: str, result, cache_dict: dict):
    """Save result to cache."""
    cache_dict[file_hash] = result

def optimize_image(file_path: str, quality: int = 85, max_size: int = 1800) -> bytes:
    """Optimize image size before sending to APIs."""
    try:
        img = Image.open(file_path)
        
        # Resize if needed
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Image optimization failed for {file_path}: {str(e)}. Using original file.")
        with open(file_path, 'rb') as f:
            return f.read()

def validate_query_result(query_text, value):
    """Validate the result of a query against its expected pattern."""
    if value == "Not Found" or not value:
        logger.info(f"Validation failed for {query_text}: Value is 'Not Found' or empty")
        return False
    
    pattern = VALIDATION_PATTERNS.get(query_text)
    if not pattern:
        logger.warning(f"No validation pattern defined for {query_text}, assuming valid")
        return True
    
    is_valid = bool(re.match(pattern, value))
    logger.info(f"Validation for {query_text}: Value='{value}', Pattern='{pattern}', Valid={is_valid}")
    return is_valid

def get_pattern_description(query_text):
    """Get a human-readable description of the validation pattern for a query."""
    if query_text == "What is the chassis number?":
        return "Alphanumeric, uppercase, 10-17 characters, NO SPACES (example: MD2A76AZ3PCJ46411)"
    elif query_text == "What is the engine number?":
        return "Alphanumeric, uppercase, 8-15 characters, NO SPACES (example: J3A5FCR1230192)"
    elif query_text == "What is the make?":
        return "Only letters and spaces (example: Honda)"
    elif query_text == "What is the model?":
        return "Alphanumeric, spaces, and hyphens (example: CB 350)"
    elif query_text == "What is the color?":
        return "Only letters and spaces (example: DEEP RED)"
    elif query_text == "What is the customer name?":
        return "Letters, spaces, dots, and apostrophes (example: JOHN SMITH)"
    return ""