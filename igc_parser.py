# This module is responsible for orchestrating the data processing workflow.
# It reads IGC files from a folder and consolidates the results into a DataFrame.

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from igc_parser import find_thermals_in_file


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
    # geopy.distance.geodesic is not a valid metric for DBSCAN. Using haversine,
    # which is a standard choice for geospatial clustering. The eps (epsilon)
    # is the maximum distance in radians between two samples for one to be considered
    # as in the neighborhood of the other.
    db = DBSCAN(eps=np.radians(radius_km / 111.32), min_samples=2, metric='haversine').fit(np.radians(coords))
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
