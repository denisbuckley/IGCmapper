# This module contains the low-level logic for parsing a single IGC file
# and finding thermals based on a set of user-defined heuristics.

import os
import numpy as np
from datetime import datetime
from geopy.distance import geodesic


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

    # Read IGC file content. A real-world scenario would require a robust IGC parser.
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        # Silently skip files that can't be read.
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
                if lat_str[7] == 'S':
                    lat = -lat

                lon = int(lon_str[:3]) + int(lon_str[3:8]) / 60000.0
                if lon_str[8] == 'W':
                    lon = -lon

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

            # Calculate distance and altitude change using geopy for better accuracy
            distance = geodesic((start_point['latitude'], start_point['longitude']),
                                (end_point['latitude'], end_point['longitude'])).meters

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
