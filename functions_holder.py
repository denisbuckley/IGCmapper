#
# This script contains the core functions for parsing IGC files, identifying
# thermals, and processing them into a pandas DataFrame. The thermal detection
# and consolidation parameters are now passed as arguments from the main script.
#
# Required libraries: pandas, numpy
#
# Install with: pip install pandas numpy
#

import os
import math
import pandas as pd
import numpy as np


# Note: User-configurable variables are no longer hard-coded here,
# but are accepted as function arguments in the functions below.

def igc_to_decimal_degrees(igc_coord):
    """
    Converts a coordinate from IGC format (DDMMmmmN/S/E/W) to decimal degrees.
    """
    direction = igc_coord[-1]
    if direction in 'NS':
        degrees = float(igc_coord[:2])
        minutes = float(igc_coord[2:-1]) / 1000.0
    else:
        degrees = float(igc_coord[:3])
        minutes = float(igc_coord[3:-1]) / 1000.0

    decimal_degrees = degrees + (minutes / 60.0)

    if direction in 'SW':
        return -decimal_degrees
    return decimal_degrees


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points on the Earth
    (given in decimal degrees) using the Haversine formula.
    Returns distance in meters.
    """
    R = 6371000  # Earth's radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def time_to_seconds(time_str):
    """
    Converts a time string in 'HHMMSS' format to total seconds from midnight.
    """
    try:
        hours = int(time_str[0:2])
        minutes = int(time_str[2:4])
        seconds = int(time_str[4:6])
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return None


def find_thermals_in_file(filepath, time_window, distance_threshold, altitude_change_threshold):
    """
    Parses a single IGC file to find and group thermal events.

    Args:
        filepath (str): Path to the IGC file.
        time_window (int): Time window in seconds for climb detection.
        distance_threshold (int): Max distance traveled in meters to be a thermal.
        altitude_change_threshold (int): Min altitude gain in meters to be a thermal.

    Returns:
        list: A list of thermal events.
        int: Total duration of the flight in seconds.
        int: Total distance of the flight in meters.
    """
    try:
        latitudes = []
        longitudes = []
        altitudes = []
        timestamps_seconds = []

        with open(filepath, 'r') as file:
            for line in file:
                record_type = line[0]
                if record_type == 'B' and len(line) >= 35:
                    try:
                        time_s = time_to_seconds(line[1:7])
                        lat = igc_to_decimal_degrees(line[7:15])
                        lon = igc_to_decimal_degrees(line[15:24])
                        alt = int(line[25:30])

                        latitudes.append(lat)
                        longitudes.append(lon)
                        altitudes.append(alt)
                        timestamps_seconds.append(time_s)

                    except (ValueError, IndexError):
                        continue

        if not latitudes:
            print(f"Warning: No valid GPS points found in {filepath}. Skipping.")
            return [], 0, 0

        # Find circling points
        circling_points_indices = []
        for i in range(time_window, len(altitudes)):
            altitude_diff = altitudes[i] - altitudes[i - time_window]
            distance_traveled = haversine_distance(
                latitudes[i], longitudes[i],
                latitudes[i - time_window], longitudes[i - time_window]
            )
            if altitude_diff > altitude_change_threshold and distance_traveled < distance_threshold:
                circling_points_indices.append(i)

        # Group successive circling points into single thermals
        thermals_data = []
        if not circling_points_indices:
            flight_duration = timestamps_seconds[-1] - timestamps_seconds[0] if len(timestamps_seconds) > 1 else 0
            flight_distance = haversine_distance(latitudes[0], longitudes[0], latitudes[-1], longitudes[-1]) if len(
                latitudes) > 1 else 0
            return [], flight_duration, flight_distance

        # Initialize the first thermal
        current_thermal = {
            'start_index': circling_points_indices[0],
            'end_index': circling_points_indices[0]
        }

        for i in range(1, len(circling_points_indices)):
            current_point_index = circling_points_indices[i]
            previous_point_index = circling_points_indices[i - 1]

            if current_point_index == previous_point_index + 1:
                # Still in the same thermal, just update the end index
                current_thermal['end_index'] = current_point_index
            else:
                # New thermal starts, so finalize the previous one
                current_thermal['altitude_gain'] = altitudes[current_thermal['end_index']] - altitudes[
                    current_thermal['start_index']]
                thermals_data.append(current_thermal)

                # Start a new thermal
                current_thermal = {
                    'start_index': current_point_index,
                    'end_index': current_point_index,
                }

        # Append the last thermal after the loop
        current_thermal['altitude_gain'] = altitudes[current_thermal['end_index']] - altitudes[
            current_thermal['start_index']]
        thermals_data.append(current_thermal)

        # Add location and climb rate to each thermal
        for thermal in thermals_data:
            start_lat = latitudes[thermal['start_index']]
            start_lon = longitudes[thermal['start_index']]
            end_lat = latitudes[thermal['end_index']]
            end_lon = longitudes[thermal['end_index']]
            thermal['start_location'] = (start_lat, start_lon)
            thermal['end_location'] = (end_lat, end_lon)

            duration = timestamps_seconds[thermal['end_index']] - timestamps_seconds[thermal['start_index']]
            thermal['climb_rate'] = thermal['altitude_gain'] / duration if duration > 0 else 0

        flight_duration = timestamps_seconds[-1] - timestamps_seconds[0] if len(timestamps_seconds) > 1 else 0
        flight_distance = 0
        for i in range(1, len(latitudes)):
            flight_distance += haversine_distance(latitudes[i - 1], longitudes[i - 1], latitudes[i], longitudes[i])

        return thermals_data, flight_duration, flight_distance
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return [], 0, 0
    except Exception as e:
        print(f"An unexpected error occurred while processing {filepath}: {e}")
        return [], 0, 0


def get_thermals_as_dataframe(folder_path, time_window, distance_threshold, altitude_change_threshold):
    """
    Analyzes IGC files in a folder and returns a pandas DataFrame with
    the start and end coordinates, and climb rate of each thermal found.

    Args:
        folder_path (str): The path to the folder containing IGC files.
        time_window (int): Time window in seconds for climb detection.
        distance_threshold (int): Max distance traveled in meters to be a thermal.
        altitude_change_threshold (int): Min altitude gain in meters to be a thermal.

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
        # Pass the user-defined parameters to find_thermals_in_file
        thermals, _, _ = find_thermals_in_file(filename, time_window, distance_threshold, altitude_change_threshold)

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


def consolidate_thermals(df, min_climb_rate, radius_km):
    """
    Filters a DataFrame of thermals and consolidates those that are within
    a specified radius, keeping only the strongest thermal in each cluster.

    Args:
        df (pd.DataFrame): DataFrame of thermal data.
        min_climb_rate (float): Minimum climb rate in m/s to be considered.
        radius_km (int): Radius in kilometers for clustering.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
               1. A DataFrame with consolidated thermals including strength.
               2. A DataFrame with only the latitude and longitude of the consolidated thermals.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Step 1: Filter out thermals below the minimum climb rate
    filtered_df = df[df['climb_rate_m_per_s'] > min_climb_rate].copy()
    if filtered_df.empty:
        print(f"No thermals found with a climb rate greater than {min_climb_rate} m/s.")
        return pd.DataFrame(), pd.DataFrame()

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
            lambda row: haversine_distance(
                strongest_lat, strongest_lon,
                row['thermal_start_lat'], row['thermal_start_lon']
            ) / 1000,  # Convert to km
            axis=1
        )

        # Mark all thermals in the cluster for removal
        thermals_to_remove_indices = filtered_df[distances <= radius_km].index

        # Remove the thermals in the cluster from the DataFrame for the next iteration
        filtered_df = filtered_df.drop(thermals_to_remove_indices).reset_index(drop=True)

    final_df_with_strength = pd.DataFrame(consolidated_thermals).reset_index(drop=True)

    # Filter and rename the columns for the DataFrame with strength
    final_df_with_strength = final_df_with_strength[
        ['thermal_start_lat', 'thermal_start_lon', 'climb_rate_m_per_s']].copy()
    final_df_with_strength = final_df_with_strength.rename(columns={
        'thermal_start_lat': 'latitude',
        'thermal_start_lon': 'longitude',
        'climb_rate_m_per_s': 'strength_m_per_s'
    })

    # Create the second DataFrame with only coordinates
    final_df_coords = final_df_with_strength[['latitude', 'longitude']].copy()

    print(f"Consolidated {len(df)} thermals into {len(final_df_with_strength)} events.")
    return final_df_with_strength, final_df_coords

