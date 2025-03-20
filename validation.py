# In validation.py

import logging
import pandas as pd

logger = logging.getLogger(__name__)

class CSVValidator:
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

    async def validate_against_csv(self, results):
        """Validate extraction results against the CSV database using multiple primary keys in sequence."""
        if self.csv_df is None:
            logger.warning("CSV not loaded")
            return {query: {"is_valid": False, "csv_value": "Not Found", "primary_key_used": None} for query in results}

        validation_results = {query: {"is_valid": False, "csv_value": "Not Found", "primary_key_used": None} for query in results}
        matching_row = None
        primary_key_used = None

        # Try each primary key in sequence
        for primary_key_query in self.primary_keys:
            primary_key_value = results.get(primary_key_query, "Not Found").lower()
            if primary_key_value == "Not Found":
                logger.info(f"Skipping {primary_key_query} as it was not found in extracted results")
                continue

            primary_key_col = self.query_to_csv_mapping.get(primary_key_query)
            if not primary_key_col or primary_key_col not in self.csv_df.columns:
                logger.error(f"Primary key column '{primary_key_col}' not found in CSV")
                continue

            matching_rows = self.csv_df[self.csv_df[primary_key_col] == primary_key_value]
            if len(matching_rows) == 1:
                matching_row = matching_rows.iloc[0]
                primary_key_used = primary_key_query
                logger.info(f"Match found using {primary_key_query} = '{primary_key_value}'")
                break
            elif len(matching_rows) > 1:
                logger.warning(f"Multiple matches for {primary_key_query}: '{primary_key_value}'. Using first match.")
                matching_row = matching_rows.iloc[0]
                primary_key_used = primary_key_query
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

            if extracted_value != "Not Found":
                extracted_value_lower = extracted_value.lower()
                if not pd.isna(csv_value):
                    # Relaxed matching for text fields
                    if query in ["What is the customer name?", "What is the make?", "What is the model?", 
                                 "What is the Vehicle_Insurance_Company or policy provider company?"]:
                        validation_results[query]["is_valid"] = (extracted_value_lower in csv_value or csv_value in extracted_value_lower)
                    else:
                        validation_results[query]["is_valid"] = (extracted_value_lower == csv_value)

            validation_results[query]["primary_key_used"] = primary_key_used
            logger.info(f"Validation for '{query}': is_valid={validation_results[query]['is_valid']}, Extracted='{extracted_value}', CSV='{validation_results[query]['csv_value']}', Primary Key='{primary_key_used}'")

        return validation_results