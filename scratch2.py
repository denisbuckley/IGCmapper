# This script analyzes multiple IGC files to determine the relationship
# between 'max_thermal_distance_km' and the number of valid thermal transitions.
# It plots the total number of thermal transitions against the distance and
# finds the plateau point where the count stabilizes.
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

# --- User-configurable variables (held constant for this analysis) ---
time_window = 30  # seconds
distance_threshold = 300  # meters, max distance traveled in the time window
altitude_change_threshold = 73  # meters
max_gap_seconds = 25  # seconds, maximum time gap to consider two segments part of the same thermal
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


def find_thermals(filepath, altitude_change_threshold, time_window, distance_threshold, max_gap_seconds):
    """
    Parses a single IGC file to find thermal segments.
    Returns a list of thermal events.
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
            return []

        sustained_lift_points_indices = []
        for i in range(time_window, len(altitudes)):
            altitude_diff = altitudes[i] - altitudes[i - time_window]
            if altitude_diff > altitude_change_threshold:
                sustained_lift_points_indices.append(i)

        thermals_data = []

        if not sustained_lift_points_indices:
            return []

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

                if distance_traveled < distance_threshold:
                    segment = {
                        'start_location': (latitudes[start_index], longitudes[start_index]),
                        'end_location': (latitudes[end_index], longitudes[end_index]),
                        'start_timestamp': timestamps_seconds[start_index],
                        'end_timestamp': timestamps_seconds[end_index],
                    }
                    thermals_data.append(segment)

                current_segment_indices = [current_point_index]

        if current_segment_indices:
            start_index = current_segment_indices[0]
            end_index = current_segment_indices[-1]
            distance_traveled = haversine_distance(
                latitudes[start_index], longitudes[start_index],
                latitudes[end_index], longitudes[end_index]
            )

            if distance_traveled < distance_threshold:
                segment = {
                    'start_location': (latitudes[start_index], longitudes[start_index]),
                    'end_location': (latitudes[end_index], longitudes[end_index]),
                    'start_timestamp': timestamps_seconds[start_index],
                    'end_timestamp': timestamps_seconds[end_index],
                }
                thermals_data.append(segment)

        return thermals_data
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while processing {filepath}: {e}")
        return []


def count_thermal_transitions(thermals_data, max_thermal_distance_km, max_gliding_time_min):
    """
    Counts the number of valid transitions between thermals.
    A transition is valid if the distance and time gap between thermals
    is below the specified thresholds.
    """
    if len(thermals_data) < 2:
        return 0

    valid_transitions = 0
    for i in range(len(thermals_data) - 1):
        thermal_1 = thermals_data[i]
        thermal_2 = thermals_data[i + 1]

        distance = haversine_distance(
            thermal_1['end_location'][0], thermal_1['end_location'][1],
            thermal_2['start_location'][0], thermal_2['start_location'][1]
        ) / 1000  # Convert to km

        time_gap_seconds = thermal_2['start_timestamp'] - thermal_1['end_timestamp']
        time_gap_minutes = time_gap_seconds / 60

        if distance <= max_thermal_distance_km and time_gap_minutes <= max_gliding_time_min:
            valid_transitions += 1

    return valid_transitions


def main():
    """
    Main function to analyze the effect of max_thermal_distance_km.
    It plots the total number of thermal transitions detected across all files
    against a range of max thermal distances and finds the plateau point.
    """
    # --- Input folder to analyze ---
    folder_path = "./igc"

    # Filter for IGC files
    igc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.igc')]

    if not igc_files:
        print(f"No IGC files found in the folder: {folder_path}. Please check the path and try again.")
        return

    # Define the range of merge distances to test
    thermal_distances_to_test = np.arange(5, 105, 5)  # From 5km to 100km, in steps of 5km
    total_transitions = []

    print("Starting analysis of thermal transitions vs. max thermal distance...")
    # Process each file individually to ensure we don't connect thermals across flights.
    for thermal_dist in thermal_distances_to_test:
        total_transitions_for_dist = 0
        for filename in igc_files:
            thermals = find_thermals(filename, altitude_change_threshold, time_window, distance_threshold,
                                     max_gap_seconds)
            transitions = count_thermal_transitions(thermals, thermal_dist, max_gliding_time_min)
            total_transitions_for_dist += transitions
        total_transitions.append(total_transitions_for_dist)
        print(f"Max Thermal Distance {thermal_dist}km: Detected {total_transitions_for_dist} transitions")

    # --- Find the plateau point ---
    plateau_distance = None
    # We look for the first point where the count doesn't increase from the previous step.
    for i in range(1, len(total_transitions)):
        if total_transitions[i] <= total_transitions[i - 1]:
            # This is the first distance at which the count stops increasing
            plateau_distance = thermal_distances_to_test[i]
            break

    if plateau_distance:
        print(
            f"\nAnalysis complete. The plateau in the thermal transitions begins at approximately {plateau_distance}km.")
    else:
        print(
            "\nAnalysis complete. A clear plateau was not found in the tested range. Consider extending the test range.")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(thermal_distances_to_test, total_transitions, marker='o', linestyle='-', color='b')
    plt.title('Total Thermal Transitions vs. Max Thermal Distance')
    plt.xlabel('Max Thermal Distance (km)')
    plt.ylabel('Total Number of Thermal Transitions')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
