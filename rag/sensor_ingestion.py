import pandas as pd
import zipfile
import io

def load_sensor_data(zip_path: str, internal_file_path: str, is_damaged: bool) -> pd.DataFrame:
    """
    Eswar's Hand-off for Kolla:
    Loads a headerless Excel file from the professor's ZIP and standardizes the columns.
    
    Args:
        zip_path: Path to the ZIP file (e.g., 'damaged_bridges.zip')
        internal_file_path: Path to the Excel file inside the zip.
        is_damaged: True if it's from the Damaged dataset (7 cols), False if Undamaged (9 cols).
    
    Returns:
        A cleaned Pandas DataFrame with standard column names.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(internal_file_path) as f:
            # Read without headers
            df = pd.read_excel(f, header=None)
            
            # Standardize columns based on Eswar's Week 5 Analysis
            if is_damaged:
                # 7 columns: Full Timestamp, Relative Time, 5 Sensors
                df.columns = [
                    "Absolute_Time", "Relative_Time_Sec",
                    "Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_5"
                ]
            else:
                # 9 columns: Hour, Min, Sec, Relative Time, 5 Sensors
                df.columns = [
                    "Hour", "Minute", "Second", "Relative_Time_Sec",
                    "Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_5"
                ]
                # Combine Hour, Min, Sec into a single Absolute Time string for consistency
                df["Absolute_Time"] = df["Hour"].astype(str) + ":" + df["Minute"].astype(str) + ":" + df["Second"].astype(str)
                df = df.drop(columns=["Hour", "Minute", "Second"])
                
                # Reorder columns to match the Damaged format
                df = df[["Absolute_Time", "Relative_Time_Sec", "Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_5"]]
            
            # Drop any entirely empty rows
            df = df.dropna(how='all')
            
            return df

if __name__ == "__main__":
    # Example usage for Kolla to test
    print("This is a data loader module prepared by Eswar for Week 5.")
    print("Import `load_sensor_data` into your preprocessing pipeline to begin!")
