# Purpose:
# This file contains the core logic for the thermal analysis. It uses the raw
# flight data to identify, filter, and consolidate thermal events based on
# user-defined parameters (time window, altitude gain, distance, etc.).
# It's the "engine" that performs the main computational tasks.
#
# Relationship:
# This module is a consumer of data from `igc_parser.py` and a provider of
# results to `run_thermal_analysis.py`. It imports functions from `igc_parser.py`
# and its main functions are called by `run_thermal_analysis.py`.

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from igc_parser import load_all_igc_data, calculate_climb_rate


def get_thermals_as_dataframe(folder_path, time_window, dist_threshold, alt_gain):
    """
    Identifies individual thermal events from the flight data and returns them as a DataFrame.
    """
    print("\nStarting raw thermal detection...")
    raw_data = load_all_igc_data(folder_path)
    if not raw_data:
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)
    df.sort_values(by='time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    thermals = []

    for i in range(len(df) - 1):
        for j in range(i + 1, len(df)):
            start_point = df.iloc[i]
            end_point = df.iloc[j]

            # Condition 1: Time window
            duration = (end_point['time'].hour * 3600 + end_point['time'].minute * 60 + end_point['time'].second) - \
                       (start_point['time'].hour * 3600 + start_point['time'].minute * 60 + start_point['time'].second)
            if duration > time_window:
                break

            # Condition 2: Altitude gain
            altitude_change = end_point['alt'] - start_point['alt']
            if altitude_change < alt_gain:
                continue

            # Condition 3: Distance threshold
            distance = geodesic((start_point['lat'], start_point['lon']), (end_point['lat'], end_point['lon'])).meters
            if distance <= dist_threshold:
                climb_rate = calculate_climb_rate(start_point['alt'], end_point['alt'], start_point['time'],
                                                  end_point['time'])
                thermals.append({
                    'start_time': start_point['time'],
                    'end_time': end_point['time'],
                    'lat': np.mean([start_point['lat'], end_point['lat']]),
                    'lon': np.mean([start_point['lon'], end_point['lon']]),
                    'climb_rate': climb_rate,
                    'altitude_gain': altitude_change,
                    'duration': duration
                })

    return pd.DataFrame(thermals)


def consolidate_thermals(thermal_df, min_climb_rate, radius_km):
    """
    Consolidates thermals using DBSCAN clustering and filters by climb rate.
    Returns a consolidated DataFrame and a DataFrame of coordinates.
    """
    if thermal_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Filter out thermals below the minimum climb rate
    filtered_df = thermal_df[thermal_df['climb_rate'] >= min_climb_rate].copy()

    if filtered_df.empty:
        print("\nNo thermals found that meet the minimum climb rate threshold.")
        return pd.DataFrame(), pd.DataFrame()

    # Use DBSCAN to cluster nearby thermals
    coords = filtered_df[['lat', 'lon']].values
    # Haversine distance is in degrees, so convert km to degrees for the eps parameter
    earth_radius_km = 6371
    epsilon = radius_km / earth_radius_km

    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    filtered_df['cluster'] = db.labels_

    # Aggregate clusters to get consolidated thermals
    consolidated_thermals = filtered_df.groupby('cluster').agg(
        lat=('lat', 'mean'),
        lon=('lon', 'mean'),
        avg_climb_rate=('climb_rate', 'mean'),
        total_altitude_gain=('altitude_gain', 'sum'),
        total_duration=('duration', 'sum'),
        num_thermals=('cluster', 'size')
    ).reset_index(drop=True)

    # Calculate "strength" for each consolidated thermal
    consolidated_thermals['strength'] = consolidated_thermals['num_thermals'] * consolidated_thermals['avg_climb_rate']
    consolidated_thermals.sort_values(by='strength', ascending=False, inplace=True)

    # Prepare a separate DataFrame with just the coordinates for mapping
    consolidated_coords = consolidated_thermals[['lat', 'lon', 'avg_climb_rate']].copy()

    return consolidated_thermals, consolidated_coords
