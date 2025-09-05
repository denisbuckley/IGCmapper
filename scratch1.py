import matplotlib.pyplot as plt
import numpy as np
import os
import math

# --- User-configurable variables ---
max_gap_seconds = 20  # seconds, maximum time gap to consider two segments part of the same thermal
altitude_change_threshold = 50  # meters, minimum altitude gain in a time window to be considered lift
time_window = 10  # seconds, the time window for checking altitude change
max_merge_distance_km = 20  # kilometers, maximum distance to merge closely spaced thermals
min_circles_threshold = 3  # The minimum number of full 360-degree circles to qualify as a thermal
circling_distance_threshold = 300  # meters, max displacement in the time window for a circling thermal


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


def is_circling(segment_points, min_circles_threshold):
    """
    Checks if a segment of flight data shows evidence of circling by summing the absolute
    heading changes. This is more robust than checking for consistent turns.
    """
    if len(segment_points) < 3:
        return False

    total_heading_change = 0

    # Track the last heading to calculate change.
    last_heading = get_heading(segment_points[0][0], segment_points[0][1],
                               segment_points[1][0], segment_points[1][1])

    # Now, check the rest of the segment.
    for i in range(2, len(segment_points)):
        lat_b, lon_b = segment_points[i - 1][0], segment_points[i - 1][1]
        lat_c, lon_c = segment_points[i][0], segment_points[i][1]

        current_heading = get_heading(lat_b, lon_b, lat_c, lon_c)
        delta_heading = (current_heading - last_heading + 180) % 360 - 180
        total_heading_change += abs(delta_heading)
        last_heading = current_heading

    return (total_heading_change / 360) >= min_circles_threshold


def find_thermals_and_sustained_lift(filepath, circling_distance_threshold, min_circles_threshold):
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
            return [], []

        circling_thermals = []
        straight_thermals = []

        thermal_start_index = None

        for i in range(time_window, len(altitudes)):
            altitude_diff = altitudes[i] - altitudes[i - time_window]

            # Check for a new thermal segment
            if altitude_diff > altitude_change_threshold:
                if thermal_start_index is None:
                    thermal_start_index = i - time_window
            else:
                # End of a thermal segment
                if thermal_start_index is not None:
                    end_index = i
                    segment_points = list(zip(
                        latitudes[thermal_start_index:end_index + 1],
                        longitudes[thermal_start_index:end_index + 1]
                    ))

                    segment = {
                        'start_location': (latitudes[thermal_start_index], longitudes[thermal_start_index]),
                        'end_location': (latitudes[end_index], longitudes[end_index]),
                        'start_timestamp': timestamps_seconds[thermal_start_index],
                        'end_timestamp': timestamps_seconds[end_index],
                        'altitude_gain': altitudes[end_index] - altitudes[thermal_start_index],
                        'climb_rate': (altitudes[end_index] - altitudes[thermal_start_index]) / (
                                timestamps_seconds[end_index] - timestamps_seconds[thermal_start_index])
                    }

                    lateral_displacement = haversine_distance(
                        segment['start_location'][0], segment['start_location'][1],
                        segment['end_location'][0], segment['end_location'][1]
                    )

                    if is_circling(segment_points,
                                   min_circles_threshold) and lateral_displacement < circling_distance_threshold:
                        circling_thermals.append(segment)
                    else:
                        straight_thermals.append(segment)

                    thermal_start_index = None

        # Process the last segment if it hasn't been closed
        if thermal_start_index is not None:
            end_index = len(altitudes) - 1
            if end_index > thermal_start_index:
                segment_points = list(zip(
                    latitudes[thermal_start_index:end_index + 1],
                    longitudes[thermal_start_index:end_index + 1]
                ))
                segment = {
                    'start_location': (latitudes[thermal_start_index], longitudes[thermal_start_index]),
                    'end_location': (latitudes[end_index], longitudes[end_index]),
                    'start_timestamp': timestamps_seconds[thermal_start_index],
                    'end_timestamp': timestamps_seconds[end_index],
                    'altitude_gain': altitudes[end_index] - altitudes[thermal_start_index],
                    'climb_rate': (altitudes[end_index] - altitudes[thermal_start_index]) / (
                            timestamps_seconds[end_index] - timestamps_seconds[thermal_start_index])
                }

                lateral_displacement = haversine_distance(
                    segment['start_location'][0], segment['start_location'][1],
                    segment['end_location'][0], segment['end_location'][1]
                )

                if is_circling(segment_points,
                               min_circles_threshold) and lateral_displacement < circling_distance_threshold:
                    circling_thermals.append(segment)
                else:
                    straight_thermals.append(segment)

        return circling_thermals, straight_thermals
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return [], []
    except Exception as e:
        print(f"An unexpected error occurred while processing {filepath}: {e}")
        return [], []


def merge_thermals(thermals, max_merge_distance_km):
    """
    Merges closely spaced thermals into single events based on a maximum distance threshold.
    """
    if not thermals:
        return []

    merged_events = []
    current_event = thermals[0]

    for i in range(1, len(thermals)):
        next_thermal = thermals[i]
        distance_between_thermals = haversine_distance(
            current_event['end_location'][0], current_event['end_location'][1],
            next_thermal['start_location'][0], next_thermal['start_location'][1]
        ) / 1000  # Convert to km

        if distance_between_thermals <= max_merge_distance_km:
            # Merge the next thermal into the current event
            current_event['end_location'] = next_thermal['end_location']
            current_event['end_timestamp'] = next_thermal['end_timestamp']
            current_event['altitude_gain'] += next_thermal['altitude_gain']
        else:
            # Start a new event
            merged_events.append(current_event)
            current_event = next_thermal

    # Add the last event
    merged_events.append(current_event)
    return merged_events


def main():
    """
    Main function to analyze the effect of a circling_distance_threshold on thermal count.
    It plots the total number of circling thermals detected across all files
    against a range of circling distances.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, "igc")

    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' was not found.")
        return

    igc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.igc')]

    if not igc_files:
        print(f"No IGC files found in the folder: {folder_path}. Please check the path and try again.")
        return

    circling_distances_to_test = np.arange(100, 501, 50)
    total_circling_thermal_counts = []

    print(f"Starting analysis of circling thermal count vs. lateral displacement threshold...")
    print(
        f"Parameters: min_circles_threshold={min_circles_threshold} circles, altitude_change_threshold={altitude_change_threshold}m, time_window={time_window}s, max_merge_distance_km={max_merge_distance_km}km")

    for circling_distance in circling_distances_to_test:
        total_thermals_for_distance = 0
        for filename in igc_files:
            circling_thermals, _ = find_thermals_and_sustained_lift(
                filename,
                circling_distance_threshold=circling_distance,
                min_circles_threshold=min_circles_threshold
            )
            merged_thermals = merge_thermals(circling_thermals, max_merge_distance_km=max_merge_distance_km)
            total_thermals_for_distance += len(merged_thermals)
            print(f"  - Processed {os.path.basename(filename)}: Found {len(merged_thermals)} thermals.")

        total_circling_thermal_counts.append(total_thermals_for_distance)
        print(
            f"\nCircling Distance Threshold {circling_distance}m: Detected a total of {total_thermals_for_distance} circling thermals")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(circling_distances_to_test, total_circling_thermal_counts, marker='o', linestyle='-', color='b')
    plt.title(
        f'Total Circling Thermals Detected vs. Lateral Displacement Threshold (min_circles_threshold = {min_circles_threshold})')
    plt.xlabel('Lateral Displacement Threshold (meters)')
    plt.ylabel('Total Number of Merged Thermal Events Detected')
    plt.grid(True)

    plt.savefig('thermal_analysis.png')
    print("\nAnalysis complete. The plot has been saved as 'thermal_analysis.png'.")
    plt.show()


if __name__ == "__main__":
    main()
