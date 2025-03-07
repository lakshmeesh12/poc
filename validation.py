import logging
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

class CSVValidator:
    def __init__(self):
        self.csv_df = None
        self.csv_to_query_mapping = {}
        self.query_to_csv_mapping = {}

    async def load_csv_database(self, csv_path):
        """Load the CSV database into a pandas DataFrame with proper handling of column names."""
        try:
            # Try standard CSV loading first
            try:
                df = pd.read_csv(csv_path)
            except:
                # If standard loading fails, try with different delimiters or custom parsing
                with open(csv_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Check if the file has proper delimiters
                if ',' not in content and ' ' in content:
                    # Treat spaces as separators, but be careful about spaces in values
                    lines = content.strip().split('\n')
                    if lines:
                        # Parse header to get column names
                        header_line = lines[0]
                        # Try to identify column names by looking for expected headers
                        expected_headers = ["Chassis No.", "Engine No.", "Make", "Colors", "Model Name", "Customer Name"]
                        
                        # Find indices of headers in the string
                        header_positions = {}
                        for header in expected_headers:
                            if header in header_line:
                                header_positions[header] = header_line.find(header)
                        
                        # Sort headers by position
                        sorted_headers = [h for h, _ in sorted(header_positions.items(), key=lambda x: x[1])]
                        
                        # Parse data rows
                        data = []
                        for i in range(1, len(lines)):
                            row_data = {}
                            line = lines[i]
                            
                            # For each header, find the data between current header position and next header position
                            for j, header in enumerate(sorted_headers):
                                start_pos = header_positions[header]
                                if j < len(sorted_headers) - 1:
                                    end_pos = header_positions[sorted_headers[j+1]]
                                    value = line[start_pos:end_pos].strip()
                                else:
                                    value = line[start_pos:].strip()
                                
                                row_data[header] = value
                            
                            data.append(row_data)
                        
                        # Create DataFrame
                        df = pd.DataFrame(data)
                    else:
                        raise ValueError("Failed to parse CSV file - no content found")
            
            # Log the column names to help with debugging
            logger.info(f"CSV columns found: {list(df.columns)}")
            
            # Define CSV to query mapping
            csv_to_query_mapping = {
                "Chassis No.": "What is the chassis number?",
                "Engine No.": "What is the engine number?",
                "Make": "What is the make?",
                "Model Name": "What is the model?",
                "Colors": "What is the color?",
                "Customer Name": "What is the customer name?"
            }
            
            # Check if expected columns exist
            missing_columns = [col for col in csv_to_query_mapping.keys() if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing expected columns in CSV: {missing_columns}")
                
                # Try to map close column names (case insensitive)
                actual_columns = [col.lower() for col in df.columns]
                fixed_mapping = {}
                
                for expected_col in csv_to_query_mapping.keys():
                    if expected_col not in df.columns:
                        expected_lower = expected_col.lower()
                        expected_no_dot = expected_lower.replace('.', '')
                        expected_no_space = expected_lower.replace(' ', '')
                        
                        # Find closest match
                        for i, actual_col in enumerate(actual_columns):
                            actual_no_dot = actual_col.replace('.', '')
                            actual_no_space = actual_col.replace(' ', '')
                            
                            if (expected_lower in actual_col or 
                                actual_col in expected_lower or
                                expected_no_dot in actual_no_dot or
                                expected_no_space in actual_no_space):
                                fixed_mapping[expected_col] = df.columns[i]
                                logger.info(f"Mapped '{expected_col}' to existing column '{df.columns[i]}'")
                                break
                
                # Update column mapping
                for expected_col, actual_col in fixed_mapping.items():
                    query = csv_to_query_mapping[expected_col]
                    csv_to_query_mapping[actual_col] = query
                    del csv_to_query_mapping[expected_col]
            
            # Set the class variables
            self.csv_to_query_mapping = csv_to_query_mapping
            self.query_to_csv_mapping = {v: k for k, v in csv_to_query_mapping.items()}
            
            logger.info(f"Updated CSV mapping: {self.csv_to_query_mapping}")

            # Convert all string columns to lowercase for case-insensitive matching
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.lower()
                    
            logger.info(f"Successfully loaded CSV database with {len(df)} records")
            self.csv_df = df
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV database: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def validate_against_csv(self, results):
        """
        Validate extraction results against the CSV database.
        Uses chassis number as the primary key.
        """
        if self.csv_df is None:
            logger.warning("CSV database not loaded, cannot validate")
            return {query: False for query in results}
            
        validation_results = {}
        
        # Get the chassis number from the results
        chassis_query = "What is the chassis number?"
        chassis_number = results.get(chassis_query, "Not Found")
        
        if chassis_number == "Not Found":
            logger.warning("Chassis number not found, cannot validate against CSV")
            return {query: False for query in results}
        
        # Find the matching row in the CSV
        chassis_col = self.query_to_csv_mapping.get(chassis_query)
        if not chassis_col or chassis_col not in self.csv_df.columns:
            logger.error(f"Cannot find chassis column '{chassis_col}' in CSV columns: {list(self.csv_df.columns)}")
            return {query: False for query in results}
        
        # Convert chassis_number to lowercase for comparison
        chassis_number = chassis_number.lower()
        
        # Find matching row
        matching_rows = self.csv_df[self.csv_df[chassis_col] == chassis_number]
        
        if len(matching_rows) == 0:
            logger.warning(f"No matching chassis number '{chassis_number}' found in CSV")
            return {query: False for query in results}
        
        if len(matching_rows) > 1:
            logger.warning(f"Multiple entries found for chassis number '{chassis_number}' in CSV")
        
        # Use the first matching row
        csv_row = matching_rows.iloc[0]
        
        # Validate each field
        for query, extracted_value in results.items():
            if query == chassis_query:
                # Chassis number already matched
                validation_results[query] = True
                continue
                
            csv_col = self.query_to_csv_mapping.get(query)
            if not csv_col or csv_col not in self.csv_df.columns:
                logger.warning(f"No CSV column found for query '{query}' (mapped to '{csv_col}')")
                validation_results[query] = False
                continue
                
            csv_value = csv_row[csv_col]
            
            # Convert extracted value to lowercase for comparison
            if extracted_value != "Not Found":
                extracted_value = extracted_value.lower()
                
            # Check if values match
            if extracted_value == "Not Found" or pd.isna(csv_value):
                validation_results[query] = False
            else:
                # Allow partial matches for certain fields (like names)
                if query in ["What is the customer name?", "What is the make?", "What is the model?"]:
                    validation_results[query] = (extracted_value in csv_value or csv_value in extracted_value)
                else:
                    validation_results[query] = (extracted_value == csv_value)
                    
            logger.info(f"Validation for '{query}': {validation_results[query]} (Extracted: '{extracted_value}', CSV: '{csv_value}')")
        
        return validation_results