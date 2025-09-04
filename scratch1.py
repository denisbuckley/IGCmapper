import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.spatial import ConvexHull

# --- User-configurable variables ---
# Heuristic parameters for identifying a sustained climb period.
# Note: 'altitude_change_threshold' is now a constant in the main function.
# New parameter to merge lift segments separated by short gaps.
max_gap_seconds = 20  # seconds, maximum time gap to consider two segments part of the same thermal
# New parameter to filter out large distances that skew the distribution.
max_thermal_distance_km = 20  # kilometers, maximum distance to consider between thermals
# New parameter to group closely spaced thermals into a single event for distance calculation.
max_merge_distance_km = 5  # kilometers, maximum distance to consider two thermals as a single event
# NEW: Filter out distances based on time. Prevents linking thermals separated by long breaks.
max_gliding_time_min = 5  # minutes, max time gap between thermals to consider for distance calculation
# NEW: Threshold for the circling check.
min_total_heading_change = 300  # degrees, the minimum total change in heading to qualify as a circle
# A new distance threshold to distinguish between tight circling thermals and linear sustained lift.
circling_distance_threshold = 300  # meters, max distance traveled in the time window for a circling thermal


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


def get_heading(lat1, lon1, lat2, lon2):
    """
    Calculates the heading (bearing) in degrees between two points.
    Returns a value from -180 to 180 degrees.
    """
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(
        lon2_rad - lon1_rad)

    heading = math.atan2(y, x)
    return math.degrees(heading)


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


def time_to_seconds(time_str):
    """
    Converts a time string in 'HHMMSS' format to total seconds from midnight.
    """
    if not time_str or len(time_str) < 6:
        return None
    try:
        hours = int(time_str[0:2])
        minutes = int(time_str[2:4])
        seconds = int(time_str[4:6])
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return None


def is_circling(segment_points):
    """
    Checks if a segment of flight data shows evidence of circling.
    A segment is considered to be circling if the total heading change is
    significant and the turns are mostly in a consistent direction.
    """
    if len(segment_points) < 3:
        return False

    total_heading_change = 0

    # Calculate initial turn direction.
    lat_p1, lon_p1 = segment_points[0][0], segment_points[0][1]
    lat_p2, lon_p2 = segment_points[1][0], segment_points[1][1]
    lat_p3, lon_p3 = segment_points[2][0], segment_points[2][1]

    heading_1 = get_heading(lat_p1, lon_p1, lat_p2, lon_p2)
    heading_2 = get_heading(lat_p2, lon_p2, lat_p3, lon_p3)
    initial_delta_heading = (heading_2 - heading_1 + 180) % 360 - 180
    turn_direction = 1 if initial_delta_heading >= 0 else -1
    total_heading_change += abs(initial_delta_heading)

    # Now, check the rest of the segment.
    for i in range(2, len(segment_points) - 1):
        lat_a, lon_a = segment_points[i - 2][0], segment_points[i - 2][1]
        lat_b, lon_b = segment_points[i - 1][0], segment_points[i - 1][1]
        lat_c, lon_c = segment_points[i][0], segment_points[i][1]

        heading_ab = get_heading(lat_a, lon_a, lat_b, lon_b)
        heading_bc = get_heading(lat_b, lon_b, lat_c, lon_c)

        delta_heading = (heading_bc - heading_ab + 180) % 360 - 180

        # Check for a consistent turn direction.
        if turn_direction * delta_heading < 0 and abs(delta_heading) > 10:  # Allow for small wobbles
            return False

        total_heading_change += abs(delta_heading)

    return total_heading_change >= min_total_heading_change


def find_thermals_and_sustained_lift(filepath, altitude_change_threshold, time_window):
    """
    Parses a single IGC file to find thermals (circling) and sustained lift (linear).
    Returns a list of circling thermal events and straight-flying thermals.
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

                        if time_s is not None:
                            latitudes.append(lat)
                            longitudes.append(lon)
                            altitudes.append(alt)
                            timestamps_seconds.append(time_s)

                    except (ValueError, IndexError):
                        continue

        if not latitudes:
            print(f"Warning: No valid GPS points found in {filepath}. Skipping.")
            return [], [], 0, []

        sustained_lift_points_indices = []
        for i in range(time_window, len(altitudes)):
            altitude_diff = altitudes[i] - altitudes[i - time_window]
            if altitude_diff > altitude_change_threshold:
                sustained_lift_points_indices.append(i)

        circling_thermals = []
        straight_thermals = []

        if not sustained_lift_points_indices:
            flight_duration = timestamps_seconds[-1] - timestamps_seconds[0] if len(timestamps_seconds) > 1 else 0
            flight_path = list(zip(latitudes, longitudes))
            return [], [], flight_duration, flight_path

        current_segment_indices = [sustained_lift_points_indices[0]]
        for i in range(1, len(sustained_lift_points_indices)):
            current_point_index = sustained_lift_points_indices[i]
            previous_point_index = sustained_lift_points_indices[i - 1]
            time_gap = timestamps_seconds[current_point_index] - timestamps_seconds[previous_point_index]

            if time_gap <= max_gap_seconds:
                current_segment_indices.append(current_point_index)
            else:
                if len(current_segment_indices) > 1:
                    start_index = current_segment_indices[0]
                    end_index = current_segment_indices[-1]

                    distance_traveled = haversine_distance(
                        latitudes[start_index], longitudes[start_index],
                        latitudes[end_index], longitudes[end_index]
                    )

                    segment_points = list(zip(
                        latitudes[start_index:end_index + 1],
                        longitudes[start_index:end_index + 1]
                    ))

                    segment = {
                        'start_location': (latitudes[start_index], longitudes[start_index]),
                        'end_location': (latitudes[end_index], longitudes[end_index]),
                        'start_timestamp': timestamps_seconds[start_index],
                        'end_timestamp': timestamps_seconds[end_index],
                        'altitude_gain': altitudes[end_index] - altitudes[start_index],
                        'climb_rate': (altitudes[end_index] - altitudes[start_index]) / (
                                timestamps_seconds[end_index] - timestamps_seconds[start_index])
                        if timestamps_seconds[end_index] > timestamps_seconds[start_index] else 0
                    }

                    if is_circling(segment_points) and distance_traveled < circling_distance_threshold:
                        circling_thermals.append(segment)
                    else:
                        straight_thermals.append(segment)

                current_segment_indices = [current_point_index]

        # Process the last segment
        if len(current_segment_indices) > 1:
            start_index = current_segment_indices[0]
            end_index = current_segment_indices[-1]

            distance_traveled = haversine_distance(
                latitudes[start_index], longitudes[start_index],
                latitudes[end_index], longitudes[end_index]
            )

            segment_points = list(zip(
                latitudes[start_index:end_index + 1],
                longitudes[start_index:end_index + 1]
            ))

            segment = {
                'start_location': (latitudes[start_index], longitudes[start_index]),
                'end_location': (latitudes[end_index], longitudes[end_index]),
                'start_timestamp': timestamps_seconds[start_index],
                'end_timestamp': timestamps_seconds[end_index],
                'altitude_gain': altitudes[end_index] - altitudes[start_index],
                'climb_rate': (altitudes[end_index] - altitudes[start_index]) / (
                        timestamps_seconds[end_index] - timestamps_seconds[start_index])
                if timestamps_seconds[end_index] > timestamps_seconds[start_index] else 0
            }

            if is_circling(segment_points) and distance_traveled < circling_distance_threshold:
                circling_thermals.append(segment)
            else:
                straight_thermals.append(segment)

        flight_duration = timestamps_seconds[-1] - timestamps_seconds[0] if len(timestamps_seconds) > 1 else 0
        flight_path = list(zip(latitudes, longitudes))

        return circling_thermals, straight_thermals, flight_duration, flight_path
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return [], [], 0, []
    except Exception as e:
        print(f"An unexpected error occurred while processing {filepath}: {e}")
        return [], [], 0, []


def main():
    """
    Main function to analyze the effect of a time_window on thermal count.
    It plots the total number of circling thermals detected across all files
    against a range of time windows.
    """
    # --- Input folder to analyze ---
    folder_path = "./igc"

    # Filter for IGC files
    igc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.igc')]

    if not igc_files:
        print(f"No IGC files found in the folder: {folder_path}. Please check the path and try again.")
        return

    # Define the constant altitude threshold
    altitude_change_threshold = 50  # meters

    # Define the range of time windows to test
    time_windows_to_test = np.arange(5, 51, 5)  # From 5s to 50s, in steps of 5s
    total_circling_thermal_counts = []

    print(f"Starting analysis of circling thermal count vs. time window (altitude threshold constant at {altitude_change_threshold}m)...")
    for time_window in time_windows_to_test:
        total_thermals_for_window = 0
        for filename in igc_files:
            circling_thermals, _, _, _ = find_thermals_and_sustained_lift(
                filename,
                altitude_change_threshold=altitude_change_threshold,
                time_window=time_window
            )
            total_thermals_for_window += len(circling_thermals)
        total_circling_thermal_counts.append(total_thermals_for_window)
        print(f"Time Window {time_window}s: Detected {total_thermals_for_window} circling thermals")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_windows_to_test, total_circling_thermal_counts, marker='o', linestyle='-', color='b')
    plt.title('Total Circling Thermals Detected vs. Time Window')
    plt.xlabel('Time Window (seconds)')
    plt.ylabel('Total Number of Circling Thermals Detected')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
