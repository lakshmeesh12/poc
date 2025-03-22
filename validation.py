import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

class EnhancedCSVValidator:
    def __init__(self):
        self.csv_df = None  # Lowercase version for matching
        self.csv_df_original = None  # Original case for display
        self.csv_to_query_mapping = {}
        self.query_to_csv_mapping = {}
        self.primary_keys = [
            "What is the chassis number?",
            "What is the engine number?",
            "What is the InsuranceNo or policy number?",
            "What is the customer name?"
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer='word',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.similarity_thresholds = {
            "What is the chassis number?": 0.9,  # High threshold for unique identifiers
            "What is the engine number?": 0.9,   # High threshold for unique identifiers
            "What is the make?": 0.7,            # Medium threshold for brand names
            "What is the model?": 0.6,           # Lower for models which may have variants
            "What is the customer name?": 0.8,   # High but not exact for names
            "What is the InsuranceNo or policy number?": 0.95,  # Very high for policy numbers
            "What is the Vehicle_Insurance_Company or policy provider company?": 0.6  # Lower for company names that might have variations
        }
        # Initialize cache for preprocessed texts
        self.preprocess_cache = {}

    async def load_csv_database(self, csv_path):
        """Load the CSV database into a pandas DataFrame."""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"CSV columns found: {list(df.columns)}")

            # Define column mappings
            csv_to_query_mapping = {
                "ChassisNumber": "What is the chassis number?",
                "EngineMotorNumber": "What is the engine number?",
                "Make": "What is the make?",
                "ModelName": "What is the model?",
                "CustomerName": "What is the customer name?",
                "InsuranceNo": "What is the InsuranceNo or policy number?",
                "Vehicle_Insurance_Company": "What is the Vehicle_Insurance_Company or policy provider company?"
            }

            # Handle missing or misnamed columns
            missing_columns = [col for col in csv_to_query_mapping.keys() if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                actual_columns = [col.lower() for col in df.columns]
                fixed_mapping = {}
                for expected_col in csv_to_query_mapping.keys():
                    if expected_col not in df.columns:
                        expected_lower = expected_col.lower().replace('.', '').replace(' ', '')
                        for i, actual_col in enumerate(actual_columns):
                            actual_no_dot = actual_col.replace('.', '').replace(' ', '')
                            if expected_lower in actual_no_dot or actual_no_dot in expected_lower:
                                fixed_mapping[expected_col] = df.columns[i]
                                logger.info(f"Mapped '{expected_col}' to '{df.columns[i]}'")
                                break
                for expected_col, actual_col in fixed_mapping.items():
                    query = csv_to_query_mapping[expected_col]
                    csv_to_query_mapping[actual_col] = query
                    del csv_to_query_mapping[expected_col]

            self.csv_to_query_mapping = csv_to_query_mapping
            self.query_to_csv_mapping = {v: k for k, v in csv_to_query_mapping.items()}
            logger.info(f"CSV mapping: {self.csv_to_query_mapping}")

            # Store original and lowercase versions
            self.csv_df_original = df.copy()  # Preserve original case
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.lower()  # Lowercase for matching
            self.csv_df = df
            logger.info(f"Loaded CSV with {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return None
    
    @lru_cache(maxsize=1024)
    def preprocess_text(self, text):
        """
        Preprocess text for better matching, with caching for performance
        """
        if pd.isna(text) or not text or text == "Not Found":
            return ""
        
        # Convert to string if not already
        text = str(text).lower()
        
        # Remove special characters but keep alphanumerics and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common words that don't add meaning
        stop_words = ['ltd', 'limited', 'co', 'company', 'inc', 'incorporated', 'corp', 'corporation']
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)
    
    def compute_similarity(self, text1, text2, query_type=None):
        """
        Compute similarity between two text strings using TF-IDF and cosine similarity
        """
        if pd.isna(text1) or pd.isna(text2) or not text1 or not text2:
            return 0.0
        
        # Use cached preprocessed text if available
        if text1 in self.preprocess_cache:
            processed_text1 = self.preprocess_cache[text1]
        else:
            processed_text1 = self.preprocess_text(text1)
            self.preprocess_cache[text1] = processed_text1
            
        if text2 in self.preprocess_cache:
            processed_text2 = self.preprocess_cache[text2]
        else:
            processed_text2 = self.preprocess_text(text2)
            self.preprocess_cache[text2] = processed_text2
            
        # For numeric/alphanumeric identifiers like chassis numbers or policy numbers
        if query_type in ["What is the chassis number?", "What is the engine number?", "What is the InsuranceNo or policy number?"]:
            # For identifiers, use exact matching but ignore spaces, dashes, and case
            norm_text1 = re.sub(r'[\s\-]', '', processed_text1)
            norm_text2 = re.sub(r'[\s\-]', '', processed_text2)
            
            # If one is a substring of the other, consider high similarity
            if norm_text1 in norm_text2 or norm_text2 in norm_text1:
                return max(len(norm_text1) / len(norm_text2), len(norm_text2) / len(norm_text1)) if norm_text1 and norm_text2 else 0.0
            
            # Calculate character-level similarity for alphanumeric IDs
            if len(norm_text1) == 0 or len(norm_text2) == 0:
                return 0.0
                
            # Character-level Levenshtein distance ratio
            from difflib import SequenceMatcher
            return SequenceMatcher(None, norm_text1, norm_text2).ratio()
        
        # For textual fields like names or companies
        if not processed_text1 or not processed_text2:
            return 0.0
            
        # Check if one is a subset of the other
        if processed_text1 in processed_text2 or processed_text2 in processed_text1:
            return 0.9  # High similarity when one is a subset of the other
            
        # Use TF-IDF vectorization for more complex text comparison
        try:
            tfidf_matrix = self.vectorizer.fit_transform([processed_text1, processed_text2])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0

    async def validate_against_csv(self, results):
        """Validate extraction results against the CSV database using multiple primary keys in sequence."""
        if self.csv_df is None:
            logger.warning("CSV not loaded")
            return {query: {"is_valid": False, "csv_value": "Not Found", "primary_key_used": None} for query in results}

        validation_results = {query: {"is_valid": False, "csv_value": "Not Found", "primary_key_used": None, "similarity_score": 0.0} for query in results}
        matching_row = None
        primary_key_used = None

        # Try each primary key in sequence
        for primary_key_query in self.primary_keys:
            primary_key_value = results.get(primary_key_query, "Not Found")
            if primary_key_value == "Not Found":
                logger.info(f"Skipping {primary_key_query} as it was not found in extracted results")
                continue

            primary_key_col = self.query_to_csv_mapping.get(primary_key_query)
            if not primary_key_col or primary_key_col not in self.csv_df.columns:
                logger.error(f"Primary key column '{primary_key_col}' not found in CSV")
                continue

            # For primary keys, use vector similarity to find best match
            best_similarity = 0.0
            best_row_idx = -1
            threshold = self.similarity_thresholds.get(primary_key_query, 0.8)
            
            for idx, row in self.csv_df.iterrows():
                csv_value = row[primary_key_col]
                similarity = self.compute_similarity(primary_key_value, csv_value, primary_key_query)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_row_idx = idx
            
            if best_similarity >= threshold and best_row_idx >= 0:
                matching_row = self.csv_df.iloc[best_row_idx]
                primary_key_used = primary_key_query
                logger.info(f"Match found using {primary_key_query} = '{primary_key_value}' with similarity {best_similarity:.2f}")
                break

        # If no match is found after all attempts
        if matching_row is None:
            logger.warning("No match found using any primary key")
            return validation_results

        # Get the original row (with original casing) for display
        original_row = self.csv_df_original.iloc[matching_row.name]

        # Validate all fields against the matched row
        for query, extracted_value in results.items():
            csv_col = self.query_to_csv_mapping.get(query)
            if not csv_col or csv_col not in self.csv_df.columns:
                logger.warning(f"No CSV column for '{query}' (mapped to '{csv_col}')")
                continue

            csv_value = matching_row[csv_col]  # Lowercase for matching
            original_csv_value = original_row[csv_col]  # Original case for display

            if pd.isna(original_csv_value):
                validation_results[query]["csv_value"] = "Not Found"
            else:
                validation_results[query]["csv_value"] = original_csv_value

            if extracted_value != "Not Found" and not pd.isna(csv_value):
                # Use vector similarity with query-specific thresholds
                similarity = self.compute_similarity(extracted_value, csv_value, query)
                threshold = self.similarity_thresholds.get(query, 0.7)
                
                validation_results[query]["is_valid"] = (similarity >= threshold)
                validation_results[query]["similarity_score"] = similarity
                
                logger.info(f"Similarity for '{query}': {similarity:.2f}, Threshold: {threshold}, Valid: {similarity >= threshold}")
            else:
                validation_results[query]["similarity_score"] = 0.0

            validation_results[query]["primary_key_used"] = primary_key_used
            logger.info(f"Validation for '{query}': is_valid={validation_results[query]['is_valid']}, Extracted='{extracted_value}', CSV='{validation_results[query]['csv_value']}', Similarity={validation_results[query]['similarity_score']:.2f}, Primary Key='{primary_key_used}'")

        return validation_results