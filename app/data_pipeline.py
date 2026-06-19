import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
try:
    import chardet  # type: ignore
except ImportError:
    chardet = None
from app.config import DATASET_DIR, REPORTS_DIR

logger = logging.getLogger("ParkTwinAI.DataPipeline")

# Setup clean logging if not configured
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class DataPipeline:
    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.raw_df = None
        self.df = None
        self.selected_file = None

    def find_csv(self):
        """Finds the CSV file inside dataset directory."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory '{self.dataset_dir}' does not exist.")
        
        csv_files = list(self.dataset_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in dataset directory '{self.dataset_dir}'.")
        
        if len(csv_files) == 1:
            self.selected_file = csv_files[0]
            logger.info(f"Automatically selected dataset file: {self.selected_file.name}")
        else:
            # Sort by size to get largest CSV file
            csv_files.sort(key=lambda x: x.stat().st_size, reverse=True)
            self.selected_file = csv_files[0]
            msg = f"Warning: Multiple CSV files found. Selected the largest: '{self.selected_file.name}'."
            logger.warning(msg)
            print(msg)
            
        return self.selected_file

    def detect_encoding(self, file_path):
        """Detect encoding of the CSV file by checking the first 20KB."""
        if chardet is None:
            logger.warning("chardet module not available. Defaulting to utf-8.")
            return 'utf-8'
            
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(20000)
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            # Fallback if encoding is not detected properly
            if not encoding:
                encoding = 'utf-8'
            logger.info(f"Detected encoding: {encoding} with confidence {result.get('confidence', 0)}")
            return encoding
        except Exception as e:
            logger.error(f"Failed to detect encoding: {e}. Defaulting to utf-8.")
            return 'utf-8'

    def infer_column_mappings(self, columns):
        """Map actual CSV columns to standard expected names using keyword matching."""
        mappings = {}
        # Expected: latitude, longitude, created_datetime, closed_datetime, police_station, junction, vehicle_type, vehicle_number
        standard_keys = {
            "latitude": ["lat", "latitude", "gps_lat"],
            "longitude": ["lon", "lng", "longitude", "gps_lon"],
            "created_datetime": ["created", "created_at", "created_datetime", "start_time", "datetime"],
            "closed_datetime": ["closed", "closed_at", "closed_datetime", "end_time", "resolved_at"],
            "police_station": ["police", "police_station", "station", "dept"],
            "junction": ["junction", "junction_name", "intersection", "crossroad"],
            "vehicle_type": ["vehicle_type", "type_of_vehicle", "category", "updated_vehicle_type"],
            "vehicle_number": ["vehicle_number", "license", "plate", "vehicle_no", "updated_vehicle_number"],
        }
        
        for std_key, keywords in standard_keys.items():
            mapped = False
            # Try exact match first (case-insensitive)
            for col in columns:
                if col.lower() == std_key.lower():
                    mappings[col] = std_key
                    mapped = True
                    break
            if mapped:
                continue
                
            # Try keyword match
            for col in columns:
                col_lower = col.lower()
                for kw in keywords:
                    if kw.lower() in col_lower:
                        mappings[col] = std_key
                        mapped = True
                        break
                if mapped:
                    break
                    
        logger.info(f"Inferred column mappings: {mappings}")
        return mappings

    def load_data(self):
        """Load, clean, and standardize the dataset."""
        file_path = self.find_csv()
        detected_enc = self.detect_encoding(file_path)
        
        # Define sequential fallback list of encodings to try
        encodings_to_try = [detected_enc]
        if detected_enc.lower() == 'ascii':
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
        else:
            encodings_to_try_lower = [e.lower() for e in encodings_to_try]
            if 'utf-8' not in encodings_to_try_lower:
                encodings_to_try.append('utf-8')
            if 'latin-1' not in encodings_to_try_lower:
                encodings_to_try.append('latin-1')
            if 'cp1252' not in encodings_to_try_lower:
                encodings_to_try.append('cp1252')
                
        logger.info(f"Reading CSV dataset with encodings try-list: {encodings_to_try}")
        
        self.raw_df = None
        for enc in encodings_to_try:
            try:
                self.raw_df = pd.read_csv(file_path, encoding=enc, low_memory=False)
                logger.info(f"Successfully loaded CSV with encoding: {enc}")
                break
            except (UnicodeDecodeError, LookupError) as e:
                logger.warning(f"Failed to read CSV with encoding {enc}: {e}. Trying next...")
                
        if self.raw_df is None:
            # Absolute fallback that never fails to decode (latin-1 matches raw bytes 0-255)
            logger.info("Falling back to latin-1 encoding...")
            self.raw_df = pd.read_csv(file_path, encoding='latin-1', low_memory=False)
            
        self.df = self.raw_df.copy()
        
        # Standardize columns
        mappings = self.infer_column_mappings(self.df.columns)
        self.df = self.df.rename(columns=mappings)
        
        # Keep only standard columns (plus 'id' and 'location' if available for displaying)
        extra_cols = []
        if 'id' in self.df.columns:
            extra_cols.append('id')
        if 'location' in self.df.columns:
            extra_cols.append('location')
            
        std_cols = ['latitude', 'longitude', 'created_datetime', 'closed_datetime', 
                    'police_station', 'junction', 'vehicle_type', 'vehicle_number']
        
        # Filter dataframe columns
        cols_to_keep = [col for col in std_cols + extra_cols if col in self.df.columns]
        self.df = self.df[cols_to_keep]
        
        # Ensure target columns exist (add NaN if missing)
        for col in std_cols:
            if col not in self.df.columns:
                self.df[col] = np.nan
        
        # Clean string columns: Trim spaces
        str_cols = ['police_station', 'junction', 'vehicle_type', 'vehicle_number']
        if 'location' in self.df.columns:
            str_cols.append('location')
            
        for col in str_cols:
            self.df[col] = self.df[col].astype(str).str.strip()
            # Replace string representations of nulls
            self.df[col] = self.df[col].replace({'nan': np.nan, 'NULL': np.nan, 'None': np.nan, '': np.nan})
            
        # Parse datetime columns
        datetime_cols = ['created_datetime', 'closed_datetime']
        for col in datetime_cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce', utc=True)
            
        # Impute missing values
        self._impute_missing_values()
        
        # Add derived duration column in minutes
        self.df['duration_minutes'] = (self.df['closed_datetime'] - self.df['created_datetime']).dt.total_seconds() / 60.0
        # If duration is negative or unrealistically small, replace with default
        self.df.loc[self.df['duration_minutes'] <= 0, 'duration_minutes'] = 30.0
        
        # Save data quality report
        self.generate_data_quality_report()
        
        return self.df

    def _impute_missing_values(self):
        """Imputes missing dataset values, specifically missing closed_datetimes."""
        # Clean latitude/longitude: drop rows where coordinates are null
        self.df = self.df.dropna(subset=['latitude', 'longitude'])
        
        # Fill categorical columns with default placeholders
        self.df['police_station'] = self.df['police_station'].fillna("Unknown Station")
        self.df['junction'] = self.df['junction'].fillna("No Junction")
        self.df['vehicle_type'] = self.df['vehicle_type'].fillna("UNKNOWN")
        self.df['vehicle_number'] = self.df['vehicle_number'].fillna("UNKNOWN")
        
        # Impute missing closed_datetime
        # If closed_datetime is missing, let's see if we have validation_timestamp or modified_datetime in raw_df
        if 'validation_timestamp' in self.raw_df.columns:
            raw_val = pd.to_datetime(self.raw_df['validation_timestamp'], errors='coerce', utc=True)
            # Match indices
            self.df['closed_datetime'] = self.df['closed_datetime'].fillna(raw_val)
            
        # Default duration assumptions by vehicle type (in hours)
        default_durations = {
            "CAR": 2.0,
            "SCOOTER": 1.0,
            "MOTOR CYCLE": 1.0,
            "MOTORCYCLE": 1.0,
            "PASSENGER AUTO": 0.5,
            "GOODS AUTO": 1.0,
            "LGV": 3.0,
            "VAN": 2.0,
            "TRUCK": 4.0,
            "BUS": 3.0,
            "TANKER": 4.0,
            "UNKNOWN": 1.5
        }
        
        # Impute remaining missing closed_datetime based on vehicle_type duration
        missing_mask = self.df['closed_datetime'].isna()
        if missing_mask.any():
            # Get default offset for each vehicle type
            offsets = self.df.loc[missing_mask, 'vehicle_type'].map(default_durations).fillna(1.5)
            # Convert to Timedelta
            timedeltas = pd.to_timedelta(offsets, unit='h')
            # Add to created_datetime
            self.df.loc[missing_mask, 'closed_datetime'] = self.df.loc[missing_mask, 'created_datetime'] + timedeltas

    def generate_data_quality_report(self):
        """Compute data quality statistics and export to output directory."""
        if self.raw_df is None or self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        report = {
            "selected_file": self.selected_file.name,
            "row_count_raw": int(len(self.raw_df)),
            "row_count_cleaned": int(len(self.df)),
            "column_count_raw": int(len(self.raw_df.columns)),
            "column_count_cleaned": int(len(self.df.columns)),
            "missing_values_raw": self.raw_df.isna().sum().to_dict(),
            "missing_values_cleaned": self.df.isna().sum().to_dict(),
            "duplicate_count_raw": int(self.raw_df.duplicated().sum()),
            "duplicate_count_cleaned": int(self.df.duplicated().sum()),
            "data_types_cleaned": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "vehicle_types_distribution": self.df['vehicle_type'].value_counts().to_dict(),
            "police_stations_distribution": self.df['police_station'].value_counts().to_dict()
        }
        
        output_path = REPORTS_DIR / "data_quality_report.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Saved data quality report to {output_path}")
        return report

if __name__ == "__main__":
    pipeline = DataPipeline()
    cleaned_df = pipeline.load_data()
    print("Cleaned Data Sample:")
    print(cleaned_df.head())
