#
# This script reads IGC files from a specified folder, uses the find_thermals_and_sustained_lift
# function from the thermal_mapper module to identify thermals, and then compiles this
# data into a pandas DataFrame for easy export and analysis.
#
# Required libraries: pandas
# Make sure your thermal_mapper.py script is in the same directory.
#
# Install with: pip install pandas
#

import os
import pandas as pd
import thermal_mapper  # Assumes thermal_mapper.py is in the same directory

def get_thermals_as_dataframe(folder_path):
    """
    Analyzes IGC files in a folder and returns a pandas DataFrame with
    the start and end coordinates, and climb rate of each thermal found.

    Args:
        folder_path (str): The path to the folder containing IGC files.

    Returns:
        pandas.DataFrame: A DataFrame with columns for thermal data.
                          Returns an empty DataFrame if no thermals are found.
    """
    # Check if the folder exists and contains IGC files
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return pd.DataFrame()

    igc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.igc')]

    if not igc_files:
        print(f"No IGC files found in the folder: {folder_path}.")
        return pd.DataFrame()

    all_thermal_data = []
    print("Extracting thermal data from IGC files...")

    # Iterate through each IGC file and extract thermal data
    for filename in igc_files:
        thermals, _, _, _ = thermal_mapper.find_thermals_and_sustained_lift(filename)

        # Append the data of each thermal to the list
        for thermal in thermals:
            # We are extracting the start and end coordinates of each thermal
            thermal_point_data = {
                'thermal_start_lat': thermal['start_location'][0],
                'thermal_start_lon': thermal['start_location'][1],
                'thermal_end_lat': thermal['end_location'][0],
                'thermal_end_lon': thermal['end_location'][1],
                'climb_rate_m_per_s': thermal['climb_rate'],
                'altitude_gain_m': thermal['altitude_gain'],
                'file_name': os.path.basename(filename)
            }
            all_thermal_data.append(thermal_point_data)

    # Create the pandas DataFrame from the collected data
    df = pd.DataFrame(all_thermal_data)

    if not df.empty:
        print(f"Successfully extracted data for {len(df)} thermals.")
    else:
        print("No thermals were identified in the provided files.")

    return df


if __name__ == "__main__":
    # Example usage: Change this path to the directory containing your IGC files
    igc_folder = "./igc"

    # Get the thermal data as a DataFrame
    thermal_df = get_thermals_as_dataframe(igc_folder)

    # Print the DataFrame to the console
    if not thermal_df.empty:
        print("\n--- Thermal Data DataFrame ---")
        print(thermal_df)

        # You can save the DataFrame to a CSV file for use in Google Earth or other tools
        output_csv = "thermal_data.csv"
        thermal_df.to_csv(output_csv, index=False)
        print(f"\nDataFrame saved to '{output_csv}'")
