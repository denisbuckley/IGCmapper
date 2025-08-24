#
# This script reads consolidated thermal coordinates from a CSV file and filters
# them to find only those that lie within a specific angular sector.
#
# The sector is defined by a start waypoint, an end waypoint, and a
# user-specified angle (default is 30 degrees either side of the heading).
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


if __name__ == "__main__":
    # --- User-configurable file and parameters ---
    thermal_file = "consolidated_thermal_coords.csv"
    sector_angle = 30  # degrees on either side of the heading

    # Check if the thermal data file exists
    if not os.path.exists(thermal_file):
        print(f"Error: The file '{thermal_file}' was not found. Please run the thermal-data-extractor.py script first.")
        exit()

    # Get the waypoints from the user
    try:
        start_lat = float(input("Enter starting waypoint latitude: "))
        start_lon = float(input("Enter starting waypoint longitude: "))
        end_lat = float(input("Enter ending waypoint latitude: "))
        end_lon = float(input("Enter ending waypoint longitude: "))
    except ValueError:
        print("\nInvalid input. Please enter valid numbers for the coordinates. Exiting.")
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

    # Calculate the target heading (bearing) of the flight path
    target_heading = calculate_bearing(start_lat, start_lon, end_lat, end_lon)
    print(f"\nTarget heading from start to end waypoint: {target_heading:.2f} degrees")

    # Calculate the lower and upper bounds of the sector
    lower_bound = (target_heading - sector_angle + 360) % 360
    upper_bound = (target_heading + sector_angle) % 360

    print(
        f"Filtering thermals within a {sector_angle * 2}-degree sector ({lower_bound:.2f} to {upper_bound:.2f} degrees).")

    # Calculate the bearing from the starting point to each thermal
    thermal_df['bearing_to_thermal'] = thermal_df.apply(
        lambda row: calculate_bearing(
            start_lat, start_lon,
            row['latitude'], row['longitude']
        ),
        axis=1
    )

    # Filter the thermals
    if lower_bound <= upper_bound:
        # Simple case: the sector does not cross the 0/360 boundary
        filtered_thermals = thermal_df[
            (thermal_df['bearing_to_thermal'] >= lower_bound) &
            (thermal_df['bearing_to_thermal'] <= upper_bound)
            ]
    else:
        # Complex case: the sector crosses the 0/360 boundary (e.g., 340 to 20)
        filtered_thermals = thermal_df[
            (thermal_df['bearing_to_thermal'] >= lower_bound) |
            (thermal_df['bearing_to_thermal'] <= upper_bound)
            ]

    # Display results
    print(f"\nFound {len(filtered_thermals)} thermals within the specified sector.")

    if not filtered_thermals.empty:
        # Sort by distance from the start point for easier viewing
        filtered_thermals['distance_from_start_km'] = filtered_thermals.apply(
            lambda row: haversine_distance(
                start_lat, start_lon,
                row['latitude'], row['longitude']
            ) / 1000,
            axis=1
        )
        filtered_thermals = filtered_thermals.sort_values(by='distance_from_start_km')

        print("\n--- Filtered Thermals ---")
        print(filtered_thermals[['latitude', 'longitude', 'distance_from_start_km', 'bearing_to_thermal']])

        # Save the results to a new CSV file
        output_file = "filtered_thermals_in_sector.csv"
        filtered_thermals.to_csv(output_file, index=False)
        print(f"\nFiltered thermals saved to '{output_file}'")
    else:
        print("No thermals were found that meet the criteria.")
