# This script analyzes multiple IGC files to determine the relationship
# between the 'max_merge_distance_km' and the number of thermals detected.
# It plots the total number of thermals against the merge distance.
#
# The 'time_window', 'distance_threshold', 'altitude_change_threshold',
# and 'max_gap_seconds' are held constant for this analysis.
#
# Required libraries: matplotlib, numpy, scipy
# Install with: pip install matplotlib numpy scipy
#

import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.spatial import ConvexHull

# --- User-configurable variables (held constant for this analysis) ---
time_window = 30  # seconds
distance_threshold = 300  # meters, max distance traveled in the time window
altitude_change_threshold = 73  # meters
max_gap_seconds = 25  # seconds, maximum time gap to consider two segments part of the same thermal
max_thermal_distance_km = 20  # kilometers, maximum distance to consider between thermals
max_gliding_time_min = 5  # minutes, max time gap between thermals to consider for distance calculation


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


def find_thermals_and_sustained_lift(filepath, altitude_change_threshold, time_window, distance_threshold,
                                     max_gap_seconds):
    """
    Parses a single IGC file to find thermals (circling) and sustained lift (linear).
    Returns a list of thermal events, sustained lift segments, and flight path coordinates.
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
            return [], [], 0, []

        sustained_lift_points_indices = []
        for i in range(time_window, len(altitudes)):
            altitude_diff = altitudes[i] - altitudes[i - time_window]
            if altitude_diff > altitude_change_threshold:
                sustained_lift_points_indices.append(i)

        thermals_data = []
        sustained_lift_data = []

        if not sustained_lift_points_indices:
            flight_duration = timestamps_seconds[-1] - timestamps_seconds[0] if len(timestamps_seconds) > 1 else 0
            flight_path = list(zip(latitudes, longitudes))
            return [], [], flight_duration, flight_path

        if not sustained_lift_points_indices:
            return [], [], 0, []

        current_segment_indices = [sustained_lift_points_indices[0]]
        for i in range(1, len(sustained_lift_points_indices)):
            current_point_index = sustained_lift_points_indices[i]
            previous_point_index = sustained_lift_points_indices[i - 1]
            time_gap = timestamps_seconds[current_point_index] - timestamps_seconds[previous_point_index]

            if time_gap <= max_gap_seconds:
                current_segment_indices.append(current_point_index)
            else:
                start_index = current_segment_indices[0]
                end_index = current_segment_indices[-1]
                distance_traveled = haversine_distance(
                    latitudes[start_index], longitudes[start_index],
                    latitudes[end_index], longitudes[end_index]
                )
                altitude_gain = altitudes[end_index] - altitudes[start_index]
                duration = timestamps_seconds[end_index] - timestamps_seconds[start_index]

                segment = {
                    'start_location': (latitudes[start_index], longitudes[start_index]),
                    'end_location': (latitudes[end_index], longitudes[end_index]),
                    'start_timestamp': timestamps_seconds[start_index],
                    'end_timestamp': timestamps_seconds[end_index],
                    'altitude_gain': altitude_gain,
                    'climb_rate': altitude_gain / duration if duration > 0 else 0
                }

                if distance_traveled < distance_threshold:
                    thermals_data.append(segment)
                else:
                    sustained_lift_data.append(segment)

                current_segment_indices = [current_point_index]

        if current_segment_indices:
            start_index = current_segment_indices[0]
            end_index = current_segment_indices[-1]
            distance_traveled = haversine_distance(
                latitudes[start_index], longitudes[start_index],
                latitudes[end_index], longitudes[end_index]
            )
            altitude_gain = altitudes[end_index] - altitudes[start_index]
            duration = timestamps_seconds[end_index] - timestamps_seconds[start_index]

            segment = {
                'start_location': (latitudes[start_index], longitudes[start_index]),
                'end_location': (latitudes[end_index], longitudes[end_index]),
                'start_timestamp': timestamps_seconds[start_index],
                'end_timestamp': timestamps_seconds[end_index],
                'altitude_gain': altitude_gain,
                'climb_rate': altitude_gain / duration if duration > 0 else 0
            }

            if distance_traveled < distance_threshold:
                thermals_data.append(segment)
            else:
                sustained_lift_data.append(segment)

        flight_duration = timestamps_seconds[-1] - timestamps_seconds[0] if len(timestamps_seconds) > 1 else 0
        flight_path = list(zip(latitudes, longitudes))

        return thermals_data, sustained_lift_data, flight_duration, flight_path
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return [], [], 0, []
    except Exception as e:
        print(f"An unexpected error occurred while processing {filepath}: {e}")
        return [], [], 0, []


def merge_thermals(thermals_data, max_merge_distance_km):
    """
    Merges closely spaced thermals into single thermal events.
    """
    if not thermals_data:
        return []

    # Sort thermals by start time to ensure correct merging order
    thermals_data.sort(key=lambda x: x['start_timestamp'])

    merged_thermals = [thermals_data[0]]
    for i in range(1, len(thermals_data)):
        current_thermal = thermals_data[i]
        last_merged_thermal = merged_thermals[-1]

        # Calculate the distance between the end of the last merged thermal
        # and the start of the current thermal.
        distance_between_thermals = haversine_distance(
            last_merged_thermal['end_location'][0], last_merged_thermal['end_location'][1],
            current_thermal['start_location'][0], current_thermal['start_location'][1]
        )

        if distance_between_thermals <= max_merge_distance_km * 1000:
            # If they are close, merge the two thermals.
            last_merged_thermal['end_location'] = current_thermal['end_location']
            last_merged_thermal['end_timestamp'] = current_thermal['end_timestamp']
            last_merged_thermal['altitude_gain'] += current_thermal['altitude_gain']
            duration = last_merged_thermal['end_timestamp'] - last_merged_thermal['start_timestamp']
            if duration > 0:
                last_merged_thermal['climb_rate'] = last_merged_thermal['altitude_gain'] / duration
            else:
                last_merged_thermal['climb_rate'] = 0
        else:
            # If they are not close, start a new merged thermal.
            merged_thermals.append(current_thermal)

    return merged_thermals


def main():
    """
    Main function to analyze the effect of max_merge_distance_km.
    It plots the total number of thermals detected across all files against a range of merge distances.
    """
    # --- Input folder to analyze ---
    folder_path = "./igc"

    # Filter for IGC files
    igc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.igc')]

    if not igc_files:
        print(f"No IGC files found in the folder: {folder_path}. Please check the path and try again.")
        return

    # Define the range of merge distances to test
    merge_distances_to_test = np.arange(0.5, 10.5, 0.5)  # From 0.5km to 10km, in steps of 0.5km
    total_thermal_counts = []

    print("Starting analysis of thermal count vs. max merge distance...")
    for merge_dist in merge_distances_to_test:
        total_thermals_for_dist = 0
        for filename in igc_files:
            thermals, _, _, _ = find_thermals_and_sustained_lift(filename, altitude_change_threshold, time_window,
                                                                 distance_threshold, max_gap_seconds)
            merged_thermals = merge_thermals(thermals, merge_dist)
            total_thermals_for_dist += len(merged_thermals)
        total_thermal_counts.append(total_thermals_for_dist)
        print(f"Merge Distance {merge_dist}km: Detected {total_thermals_for_dist} thermals")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(merge_distances_to_test, total_thermal_counts, marker='o', linestyle='-', color='b')
    plt.title('Total Thermals Detected vs. Max Merge Distance')
    plt.xlabel('Max Merge Distance (km)')
    plt.ylabel('Total Number of Thermals Detected')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()
