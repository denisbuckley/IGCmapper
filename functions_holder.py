import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.cluster import DBSCAN


# Install with: pip install scikit-learn geopy
# You will need to install these new dependencies for the new code to work.

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
        # print(f"Error reading {file_path}: {e}")
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
                    'start_lat': start_point['latitude'],
                    'start_lon': start_point['longitude'],
                    'altitude_change': altitude_change,
                    'duration_s': time_diff,
                    'avg_climb_rate_mps': altitude_change / time_diff if time_diff > 0 else 0
                })

    return thermals


def get_thermals_as_dataframe(igc_folder, time_window, distance_threshold, altitude_change_threshold):
    """
    Reads all IGC files in a folder, finds thermals in each, and consolidates the
    results into a single pandas DataFrame. A progress bar is shown.

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

    # Use tqdm to create a progress bar for the file loop
    for file_name in tqdm(igc_files, desc="Processing IGC files"):
        file_path = os.path.join(igc_folder, file_name)
        # Pass the parameters directly to the worker function
        file_thermals = find_thermals_in_file(file_path, time_window, distance_threshold, altitude_change_threshold)
        all_thermals.extend(file_thermals)

    return pd.DataFrame(all_thermals)


def consolidate_thermals(df, min_climb_rate, radius_km):
    """
    Filters and groups thermals to find the strongest ones in a given radius.
    This version uses DBSCAN for vastly improved performance.

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
        print("No thermals met the minimum climb rate threshold. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    # Prepare data for DBSCAN
    coords = filtered_df[['start_lat', 'start_lon']].to_numpy()

    # Use geodesic distance in meters for DBSCAN, which is much more accurate
    # than Euclidean distance on lat/lon coordinates.
    db = DBSCAN(eps=radius_km / 6371, min_samples=2, metric='haversine').fit(np.radians(coords))
    labels = db.labels_

    # Create a new DataFrame with cluster labels
    clustered_df = filtered_df.copy()
    clustered_df['cluster'] = labels

    # Remove noise points (labeled as -1 by DBSCAN)
    clustered_df = clustered_df[clustered_df['cluster'] != -1]

    if clustered_df.empty:
        print("DBSCAN found no valid clusters. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    # Group by cluster and calculate consolidated thermal properties
    consolidated_df = clustered_df.groupby('cluster').agg(
        latitude=('start_lat', 'mean'),
        longitude=('start_lon', 'mean'),
        avg_climb_rate_mps=('avg_climb_rate_mps', 'mean'),
        count=('file_name', 'count')
    ).reset_index()

    # Create the coordinates DataFrame for plotting
    coords_df = consolidated_df[['latitude', 'longitude', 'avg_climb_rate_mps']].copy()

    # The strength metric for plotting should be the average climb rate
    coords_df.rename(columns={'avg_climb_rate_mps': 'strength'}, inplace=True)

    # Print a summary of the clustering results
    n_clusters = len(consolidated_df)
    n_noise = len(df) - len(clustered_df)
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")

    return consolidated_df, coords_df
