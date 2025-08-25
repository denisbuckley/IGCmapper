#
# This script reads consolidated thermal coordinates from a CSV file and filters
# them using a two-step process:
# 1. First, it filters for thermals that lie within the circular area defined
#    by the distance of the direct glidepath from start to end.
# 2. Second, it filters those remaining thermals to find those within a
#    specified angular sector from the start waypoint.
#
# Required libraries: pandas, numpy
# Make sure your functions_holder.py script is in the same directory.
#
# Install with: pip install pandas numpy
#

import pandas as pd
import numpy as np
import math
import os
from functions_holder import haversine_distance


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the initial bearing from point 1 to point 2.
    The bearing is the angle in degrees clockwise from North.

    Args:
        lat1, lon1 (float): Latitude and longitude of the starting point.
        lat2, lon2 (float): Latitude and longitude of the destination point.

    Returns:
        float: The bearing in degrees.
    """
    # Convert latitude and longitude to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate the difference in longitudes
    delta_lon = lon2_rad - lon1_rad

    # Calculate the bearing
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)

    initial_bearing = math.atan2(y, x)

    # Convert the bearing from radians to degrees and normalize to 0-360
    initial_bearing = math.degrees(initial_bearing)
    final_bearing = (initial_bearing + 360) % 360

    return final_bearing


def is_between_bearings(lower_bound, upper_bound, bearing_to_check):
    """
    Checks if a bearing is within a sector defined by two other bearings.
    This function correctly handles the 0/360 degree wrap-around.

    Args:
        lower_bound (float): The lower bound of the sector.
        upper_bound (float): The upper bound of the sector.
        bearing_to_check (float): The bearing of the point to check.

    Returns:
        bool: True if the bearing is within the sector, False otherwise.
    """
    # Normalize all bearings to be in the range [0, 360)
    lower_bound = (lower_bound + 360) % 360
    upper_bound = (upper_bound + 360) % 360
    bearing_to_check = (bearing_to_check + 360) % 360

    if lower_bound <= upper_bound:
        return lower_bound <= bearing_to_check <= upper_bound
    else:
        # The sector wraps around 0 degrees, e.g., from 350 to 10
        return bearing_to_check >= lower_bound or bearing_to_check <= upper_bound


if __name__ == "__main__":
    # --- User-configurable file and parameters ---
    thermal_file = "consolidated_thermal_coords.csv"
    default_sector_angle = 30  # degrees on either side of the heading

    # Default coordinates for convenience
    default_start_lat = -31.66
    default_start_lon = 117.24
    default_end_lat = -30.596
    default_end_lon = 116.772

    # Check if the thermal data file exists
    if not os.path.exists(thermal_file):
        print(f"Error: The file '{thermal_file}' was not found. Please run the thermal-data-extractor.py script first.")
        exit()

    # Get the waypoints and sector angle from the user
    try:
        start_lat_input = input(f"Enter starting waypoint latitude (default: {default_start_lat}): ")
        start_lat = float(start_lat_input) if start_lat_input else default_start_lat

        start_lon_input = input(f"Enter starting waypoint longitude (default: {default_start_lon}): ")
        start_lon = float(start_lon_input) if start_lon_input else default_start_lon

        end_lat_input = input(f"Enter ending waypoint latitude (default: {default_end_lat}): ")
        end_lat = float(end_lat_input) if end_lat_input else default_end_lat

        end_lon_input = input(f"Enter ending waypoint longitude (default: {default_end_lon}): ")
        end_lon = float(end_lon_input) if end_lon_input else default_end_lon

        sector_angle_input = input(f"Enter search sector degrees (default: {default_sector_angle}): ")
        sector_angle = float(sector_angle_input) if sector_angle_input else default_sector_angle
    except ValueError:
        print("\nInvalid input. Please enter valid numbers for the coordinates and angle. Exiting.")
        exit()

    # Load the thermal data
    try:
        thermal_df = pd.read_csv(thermal_file)
        if thermal_df.empty:
            print(f"The file '{thermal_file}' is empty. No thermals to analyze.")
            exit()
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        exit()

    # Calculate the glidepath distance and bearing from start to end
    glidepath_distance_km = haversine_distance(start_lat, start_lon, end_lat, end_lon) / 1000
    glidepath_bearing = calculate_bearing(start_lat, start_lon, end_lat, end_lon)

    print(f"\nGlidepath distance from start to end: {glidepath_distance_km:.2f} km")
    print(f"Glidepath bearing: {glidepath_bearing:.2f} degrees")
    print(f"Filtering thermals within a radius of {glidepath_distance_km:.2f} km from the start.")
    print(f"Then filtering for thermals within a {sector_angle * 2}-degree sector around the glidepath bearing.")

    # Calculate the bearing and distance from the starting point to each thermal
    thermal_df['bearing_to_thermal'] = thermal_df.apply(
        lambda row: calculate_bearing(
            start_lat, start_lon,
            row['latitude'], row['longitude']
        ),
        axis=1
    )
    thermal_df['distance_from_start_km'] = thermal_df.apply(
        lambda row: haversine_distance(
            start_lat, start_lon,
            row['latitude'], row['longitude']
        ) / 1000,
        axis=1
    )

    # --- Step 1: Filter by distance ---
    is_within_distance = thermal_df['distance_from_start_km'] <= glidepath_distance_km
    distance_filtered_thermals = thermal_df[is_within_distance].copy()

    # Check if any thermals passed the first filter
    if distance_filtered_thermals.empty:
        print("\nNo thermals were found within the specified glidepath distance.")
        exit()

    # --- Step 2: Filter the remaining thermals by bearing ---
    lower_bound = glidepath_bearing - sector_angle
    upper_bound = glidepath_bearing + sector_angle

    is_within_bearing_sector = distance_filtered_thermals['bearing_to_thermal'].apply(
        lambda bearing: is_between_bearings(lower_bound, upper_bound, bearing)
    )

    filtered_thermals = distance_filtered_thermals[is_within_bearing_sector]

    # Display results
    print(f"\nFound {len(filtered_thermals)} thermals that meet both criteria.")

    if not filtered_thermals.empty:
        filtered_thermals = filtered_thermals.sort_values(by='distance_from_start_km')

        print("\n--- Filtered Thermals ---")
        print(filtered_thermals[['latitude', 'longitude', 'distance_from_start_km', 'bearing_to_thermal']])

        # Save the results to a new CSV file
        output_file = "filtered_thermals_in_sector.csv"
        filtered_thermals.to_csv(output_file, index=False)
        print(f"\nFiltered thermals saved to '{output_file}'")
    else:
        print("No thermals were found that meet the criteria.")

