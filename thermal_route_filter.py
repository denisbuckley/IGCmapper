#
# This script reads consolidated thermal coordinates from a CSV file and filters
# them to find those that lie within a specified angular search arc from
# both a start and end waypoint, creating a defined flight corridor.
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
    default_sector_angle = 30  # degrees on either side of the heading

    # Check if the thermal data file exists
    if not os.path.exists(thermal_file):
        print(f"Error: The file '{thermal_file}' was not found. Please run the thermal-data-extractor.py script first.")
        exit()

    # Get the waypoints and sector angle from the user
    try:
        start_lat = float(input("Enter starting waypoint latitude: "))
        start_lon = float(input("Enter starting waypoint longitude: "))
        end_lat = float(input("Enter ending waypoint latitude: "))
        end_lon = float(input("Enter ending waypoint longitude: "))

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

    # Calculate the bearings for the flight path in both directions
    heading_start_to_end = calculate_bearing(start_lat, start_lon, end_lat, end_lon)
    heading_end_to_start = calculate_bearing(end_lat, end_lon, start_lat, start_lon)

    print(f"\nHeading from start to end: {heading_start_to_end:.2f} degrees")
    print(f"Heading from end to start: {heading_end_to_start:.2f} degrees")

    # Calculate the sector bounds from the start and end waypoints
    lower_bound_start = (heading_start_to_end - sector_angle + 360) % 360
    upper_bound_start = (heading_start_to_end + sector_angle) % 360

    lower_bound_end = (heading_end_to_start - sector_angle + 360) % 360
    upper_bound_end = (heading_end_to_start + sector_angle) % 360

    print(f"Filtering thermals within two {sector_angle * 2}-degree sectors along the flight path.")

    # Calculate the bearing from each waypoint to each thermal
    thermal_df['bearing_to_thermal_from_start'] = thermal_df.apply(
        lambda row: calculate_bearing(
            start_lat, start_lon,
            row['latitude'], row['longitude']
        ),
        axis=1
    )
    thermal_df['bearing_to_thermal_from_end'] = thermal_df.apply(
        lambda row: calculate_bearing(
            end_lat, end_lon,
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

    # Filter the thermals
    # Check if thermal bearing from START is within its sector
    if lower_bound_start <= upper_bound_start:
        is_in_start_sector = (thermal_df['bearing_to_thermal_from_start'] >= lower_bound_start) & \
                             (thermal_df['bearing_to_thermal_from_start'] <= upper_bound_start)
    else:
        is_in_start_sector = (thermal_df['bearing_to_thermal_from_start'] >= lower_bound_start) | \
                             (thermal_df['bearing_to_thermal_from_start'] <= upper_bound_start)

    # Check if thermal bearing from END is within its sector
    if lower_bound_end <= upper_bound_end:
        is_in_end_sector = (thermal_df['bearing_to_thermal_from_end'] >= lower_bound_end) & \
                           (thermal_df['bearing_to_thermal_from_end'] <= upper_bound_end)
    else:
        is_in_end_sector = (thermal_df['bearing_to_thermal_from_end'] >= lower_bound_end) | \
                           (thermal_df['bearing_to_thermal_from_end'] <= upper_bound_end)

    # Combine both filters
    filtered_thermals = thermal_df[is_in_start_sector & is_in_end_sector].copy()

    # Display results
    print(f"\nFound {len(filtered_thermals)} thermals within the specified corridor.")

    if not filtered_thermals.empty:
        filtered_thermals = filtered_thermals.sort_values(by='distance_from_start_km')

        print("\n--- Filtered Thermals ---")
        print(filtered_thermals[['latitude', 'longitude', 'distance_from_start_km', 'bearing_to_thermal_from_start',
                                 'bearing_to_thermal_from_end']])

        # Save the results to a new CSV file
        output_file = "filtered_thermals_in_sector.csv"
        filtered_thermals.to_csv(output_file, index=False)
        print(f"\nFiltered thermals saved to '{output_file}'")
    else:
        print("No thermals were found that meet the criteria.")
