#
# This script reads IGC files from a specified folder, uses the find_thermals_and_sustained_lift
# function from the thermal_mapper module to identify thermals, and then compiles this
# data into a pandas DataFrame for easy export and analysis.
#
# It has been amended to filter out thermals below a certain strength and to consolidate
# closely spaced thermals.
#
# Required libraries: pandas, numpy
# Make sure your thermal_mapper.py script is in the same directory.
#
# Install with: pip install pandas numpy
# adjust the min_climb_rate and radius_km parameters in the consolidate_thermals function call to fine-tune the filtering.

import os
import pandas as pd
import numpy as np
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


def consolidate_thermals(df, min_climb_rate=0.5, radius_km=1):
    """
    Filters a DataFrame of thermals and consolidates those that are within
    a specified radius, keeping only the strongest thermal in each cluster.

    Args:
        df (pd.DataFrame): DataFrame of thermal data.
        min_climb_rate (float): Minimum climb rate in m/s to be considered.
        radius_km (int): Radius in kilometers for clustering.

    Returns:
        pd.DataFrame: A new DataFrame with consolidated thermal data.
    """
    if df.empty:
        return pd.DataFrame()

    # Step 1: Filter out thermals below the minimum climb rate
    filtered_df = df[df['climb_rate_m_per_s'] > min_climb_rate].copy()
    if filtered_df.empty:
        print(f"No thermals found with a climb rate greater than {min_climb_rate} m/s.")
        return pd.DataFrame()

    # Sort the DataFrame by climb rate in descending order to prioritize strongest thermals
    filtered_df = filtered_df.sort_values(by='climb_rate_m_per_s', ascending=False).reset_index(drop=True)

    consolidated_thermals = []

    # Step 2: Iteratively consolidate thermals
    while not filtered_df.empty:
        # The first thermal in the sorted list is the strongest one remaining
        strongest_thermal = filtered_df.iloc[0]

        # Add this strongest thermal to our final list
        consolidated_thermals.append(strongest_thermal)

        # Create a boolean mask to identify thermals within the consolidation radius
        strongest_lat = strongest_thermal['thermal_start_lat']
        strongest_lon = strongest_thermal['thermal_start_lon']

        distances = filtered_df.apply(
            lambda row: thermal_mapper.haversine_distance(
                strongest_lat, strongest_lon,
                row['thermal_start_lat'], row['thermal_start_lon']
            ) / 1000,  # Convert to km
            axis=1
        )

        # Mark all thermals in the cluster for removal
        thermals_to_remove_indices = filtered_df[distances <= radius_km].index

        # Remove the thermals in the cluster from the DataFrame for the next iteration
        filtered_df = filtered_df.drop(thermals_to_remove_indices).reset_index(drop=True)

    final_df = pd.DataFrame(consolidated_thermals).reset_index(drop=True)

    # Filter and rename the columns as requested by the user
    final_df = final_df[['thermal_start_lat', 'thermal_start_lon', 'climb_rate_m_per_s']].copy()
    final_df = final_df.rename(columns={
        'thermal_start_lat': 'latitude',
        'thermal_start_lon': 'longitude',
        'climb_rate_m_per_s': 'strength_m_per_s'
    })

    print(f"Consolidated {len(df)} thermals into {len(final_df)} events.")
    return final_df


if __name__ == "__main__":
    # Example usage: Change this path to the directory containing your IGC files
    igc_folder = "./igc"

    # Step 1: Get the raw thermal data as a DataFrame
    thermal_df = get_thermals_as_dataframe(igc_folder)

    # Step 2: Consolidate the thermals based on the new logic
    consolidated_df = consolidate_thermals(thermal_df)

    # Print the DataFrame to the console
    if not consolidated_df.empty:
        print("\n--- Consolidated Thermal Data DataFrame ---")
        print(consolidated_df)

        # You can save the DataFrame to a CSV file for use in Google Earth or other tools
        output_csv = "consolidated_thermal_data.csv"
        consolidated_df.to_csv(output_csv, index=False)
        print(f"\nConsolidated DataFrame saved to '{output_csv}'")
