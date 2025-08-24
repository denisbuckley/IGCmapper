#
# This script contains the core functions for parsing IGC files and identifying
# thermals. It is intended to be imported by other scripts, such as the main
# analysis script.
#
# It has been created by moving functions from the original thermal_mapper.py.
#
# Required libraries: numpy
#
import math
import numpy as np

# --- User-configurable variables ---
# Heuristic parameters for identifying a thermal (a sustained circling period).
time_window = 10  # seconds to check for sustained climb and confined area
distance_threshold = 100  # meters, max distance traveled in the time window
altitude_change_threshold = 20  # meters

# Threshold for identifying a significant, continuous climb for the plot.
significant_climb_threshold = 200  # meters


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


def find_thermals_in_file(filepath):
    """
    Parses a single IGC file to find and group thermal events.
    Returns a list of thermal events, where each event is a dictionary,
    the total duration of the flight in seconds, and the total distance in meters.
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
