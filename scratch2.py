#
# This script reads a CSV file containing thermal coordinates, filters them
# to find only those within a specified radius of a given point, and
# prints the resulting data.
#
# Required libraries: pandas, geopy
#
# Install with: pip install pandas geopy
#

import pandas as pd
from geopy.distance import geodesic
import os


def filter_thermals_by_radius(file_path, center_coords, radius_km):
    """
    Reads a CSV file, filters thermal locations by a specified radius from
    a central point, and returns a new DataFrame with the filtered data.

    Args:
        file_path (str): The path to the input CSV file.
        center_coords (tuple): A tuple (latitude, longitude) of the center point.
        radius_km (float): The radius in kilometers to filter by.

    Returns:
        pandas.DataFrame: A DataFrame containing thermals within the specified radius.
                          Returns an empty DataFrame if the file is not found or
                          no thermals match the criteria.
    """
    print(f"Reading thermal data from '{file_path}'...")
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

    print(f"Filtering thermals within a {radius_km} km radius of {center_coords}...")

    # Calculate the distance for each thermal to the center point
    # We use the 'thermal_start_lat' and 'thermal_start_lon' columns
    def calculate_distance(row):
        thermal_coords = (row['thermal_start_lat'], row['thermal_start_lon'])
        return geodesic(center_coords, thermal_coords).km

    df['distance_km'] = df.apply(calculate_distance, axis=1)

    # Filter the DataFrame to keep only thermals within the radius
    filtered_df = df[df['distance_km'] <= radius_km].copy()

    print(f"\nFound {len(filtered_df)} thermals within the specified radius.")

    # Sort the results by distance for a cleaner output
    filtered_df.sort_values(by='distance_km', inplace=True)

    return filtered_df


# Main script execution
if __name__ == "__main__":
    # Define the target coordinates and radius
    target_latitude = -31.66
    target_longitude = 117.24
    filter_radius_km = 50

    center_of_interest = (target_latitude, target_longitude)
    input_csv_file = "consolidated_thermal_coords.csv"

    # Call the filtering function
    filtered_thermals_df = filter_thermals_by_radius(
        input_csv_file,
        center_of_interest,
        filter_radius_km
    )

    # Print the results
    if not filtered_thermals_df.empty:
        print("\n--- Filtered Thermal Data ---")
        print(filtered_thermals_df[['file_name', 'thermal_start_lat', 'thermal_start_lon', 'distance_km']])

        # You can save the filtered data to a new CSV file if needed
        output_csv = "filtered_thermals.csv"
        filtered_thermals_df.to_csv(output_csv, index=False)
        print(f"\nFiltered data also saved to '{output_csv}'")
    else:
        print("\nNo thermals were found within the specified radius.")

