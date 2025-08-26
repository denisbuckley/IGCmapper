import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def find_thermals_in_file(file_path, time_window, distance_threshold, altitude_change_threshold):
    """
    Analyzes an IGC file to find thermals based on user-defined heuristics.

    Args:
        file_path (str): The path to the IGC file.
        time_window (int): The time window in seconds for analysis.
        distance_threshold (int): The maximum distance in meters for a thermal.
        altitude_change_threshold (int): The minimum altitude gain in meters for a thermal.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected thermal.
    """
    thermals = []

    # Read IGC file content (this is a simplified example)
    # A real-world scenario would require a robust IGC parser.
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return thermals

    points = []
    # Simplified parsing for B-records (time, lat, lon, alt)
    for line in lines:
        if line.startswith('B'):
            try:
                # Example B-record: B0950345100694N00647355EA0037300375
                time_str = line[1:7]
                lat_str = line[7:15]
                lon_str = line[15:24]
                # GPS altitude (meters) is at index 30-35
                gps_alt = int(line[30:35])

                # Convert lat/lon strings to degrees
                lat = int(lat_str[:2]) + int(lat_str[2:7]) / 60000.0
                if lat_str[7] == 'S': lat = -lat

                lon = int(lon_str[:3]) + int(lon_str[3:8]) / 60000.0
                if lon_str[8] == 'W': lon = -lon

                time = datetime.strptime(time_str, '%H%M%S').time()

                points.append({
                    'time': time,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': gps_alt
                })
            except (ValueError, IndexError):
                continue  # Skip invalid lines

    if not points:
        return thermals

    # Heuristic-based thermal detection
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            start_point = points[i]
            end_point = points[j]

            time_diff = (datetime.combine(datetime.min, end_point['time']) -
                         datetime.combine(datetime.min, start_point['time'])).total_seconds()

            if time_diff > time_window:
                break  # Time window exceeded, move to the next start point

            # Calculate distance and altitude change
            lat1, lon1 = np.radians(start_point['latitude']), np.radians(start_point['longitude'])
            lat2, lon2 = np.radians(end_point['latitude']), np.radians(end_point['longitude'])

            # Haversine formula for distance
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            R = 6371000  # Radius of Earth in meters
            distance = R * c

            altitude_change = end_point['altitude'] - start_point['altitude']

            # Check if it's a thermal based on heuristics
            if distance <= distance_threshold and altitude_change >= altitude_change_threshold:
                thermals.append({
                    'file_name': os.path.basename(file_path),
                    'start_time': start_point['time'],
                    'end_time': end_point['time'],
                    'start_lat': start_point['latitude'],
                    'start_lon': start_point['longitude'],
                    'end_lat': end_point['latitude'],
                    'end_lon': end_point['longitude'],
                    'altitude_change': altitude_change,
                    'duration_s': time_diff,
                    'avg_climb_rate_mps': altitude_change / time_diff if time_diff > 0 else 0
                })

    return thermals


def get_thermals_as_dataframe(igc_folder, time_window, distance_threshold, altitude_change_threshold):
    """
    Reads all IGC files in a folder, finds thermals in each, and consolidates the
    results into a single pandas DataFrame.

    Args:
        igc_folder (str): The path to the folder containing IGC files.
        time_window (int): The time window in seconds for thermal detection.
        distance_threshold (int): The maximum distance in meters for a thermal.
        altitude_change_threshold (int): The minimum altitude gain in meters for a thermal.

    Returns:
        pandas.DataFrame: A DataFrame containing all detected thermals.
    """
    all_thermals = []

    if not os.path.exists(igc_folder):
        print(f"The specified folder does not exist: {igc_folder}")
        return pd.DataFrame()

    igc_files = [f for f in os.listdir(igc_folder) if f.endswith('.igc')]

    if not igc_files:
        print(f"No .igc files found in {igc_folder}.")
        return pd.DataFrame()

    print(f"Found {len(igc_files)} IGC files to process.")

    for file_name in igc_files:
        file_path = os.path.join(igc_folder, file_name)
        # Pass the parameters directly to the worker function
        file_thermals = find_thermals_in_file(file_path, time_window, distance_threshold, altitude_change_threshold)
        all_thermals.extend(file_thermals)

    return pd.DataFrame(all_thermals)


def consolidate_thermals(df, min_climb_rate, radius_km):
    """
    Filters and groups thermals to find the strongest ones in a given radius.

    Args:
        df (pandas.DataFrame): The DataFrame of thermals.
        min_climb_rate (float): Minimum average climb rate to filter thermals.
        radius_km (float): Radius in km to group thermals.

    Returns:
        tuple: A tuple containing two DataFrames:
               - The consolidated DataFrame with strength metrics.
               - A DataFrame with just coordinates for plotting.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Filter out thermals that don't meet the minimum climb rate
    filtered_df = df[df['avg_climb_rate_mps'] >= min_climb_rate].copy()

    if filtered_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Convert radius_km to meters
    radius_m = radius_km * 1000

    # Group nearby thermals (simplified clustering logic)
    consolidated_groups = []
    while not filtered_df.empty:
        # Pick the first thermal as the cluster center
        center_thermal = filtered_df.iloc[0]

        # Find all thermals within the radius of the center
        lat1, lon1 = np.radians(center_thermal['start_lat']), np.radians(center_thermal['start_lon'])
        dists = []
        for _, row in filtered_df.iterrows():
            lat2, lon2 = np.radians(row['start_lat']), np.radians(row['start_lon'])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            R = 6371000
            distance = R * c
            dists.append(distance)

        nearby_indices = [i for i, d in enumerate(dists) if d <= radius_m]
        nearby_thermals = filtered_df.iloc[nearby_indices]

        # Consolidate the group
        consolidated_group = {
            'latitude': nearby_thermals['start_lat'].mean(),
            'longitude': nearby_thermals['start_lon'].mean(),
            'avg_climb_rate_mps': nearby_thermals['avg_climb_rate_mps'].mean(),
            'count': len(nearby_thermals)
        }
        consolidated_groups.append(consolidated_group)

        # Remove the thermals in this group from the DataFrame
        filtered_df = filtered_df.drop(nearby_thermals.index).reset_index(drop=True)

    consolidated_df = pd.DataFrame(consolidated_groups)

    # Create the coordinates DataFrame for plotting
    coords_df = consolidated_df[['latitude', 'longitude', 'count']].copy()
    coords_df.rename(columns={'count': 'strength'}, inplace=True)

    return consolidated_df, coords_df
