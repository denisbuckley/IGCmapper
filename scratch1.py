import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.spatial import ConvexHull

# --- User-configurable variables ---
# Heuristic parameters for identifying a thermal (a sustained circling period).
time_window = 30  # seconds to check for sustained climb and confined area
distance_threshold = 500  # meters, max distance traveled in the time window
altitude_change_threshold = 20  # meters
# New parameter to merge thermal segments separated by short gaps.
max_gap_seconds = 20  # seconds, maximum time gap to consider two segments part of the same thermal
# New parameter to filter out large distances that skew the distribution.
max_thermal_distance_km = 20  # kilometers, maximum distance to consider between thermals
# New parameter to group closely spaced thermals into a single event for distance calculation.
max_merge_distance_km = 5  # kilometers, maximum distance to consider two thermals as a single event
# NEW: Filter out distances based on time. Prevents linking thermals separated by long breaks.
max_gliding_time_min = 5  # minutes, max time gap between thermals to consider for distance calculation
# NEW: Threshold for the circling check.
min_total_heading_change = 300  # degrees, the minimum total change in heading to qualify as a circle


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


def find_thermals_and_sustained_lift(filepath):
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

        thermals_data = []
        sustained_lift_data = []

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

                    # INLINED CIRCLING CHECK LOGIC
                    is_circling = False
                    if len(segment_points) >= 3:
                        total_heading_change = 0
                        lat_p1, lon_p1 = segment_points[0][0], segment_points[0][1]
                        lat_p2, lon_p2 = segment_points[1][0], segment_points[1][1]
                        lat_p3, lon_p3 = segment_points[2][0], segment_points[2][1]

                        heading_1 = get_heading(lat_p1, lon_p1, lat_p2, lon_p2)
                        heading_2 = get_heading(lat_p2, lon_p2, lat_p3, lon_p3)
                        initial_delta_heading = (heading_2 - heading_1 + 180) % 360 - 180
                        turn_direction = 1 if initial_delta_heading >= 0 else -1
                        total_heading_change += abs(initial_delta_heading)

                        for j in range(2, len(segment_points) - 1):
                            lat_a, lon_a = segment_points[j - 2][0], segment_points[j - 2][1]
                            lat_b, lon_b = segment_points[j - 1][0], segment_points[j - 1][1]
                            lat_c, lon_c = segment_points[j][0], segment_points[j][1]

                            heading_ab = get_heading(lat_a, lon_a, lat_b, lon_b)
                            heading_bc = get_heading(lat_b, lon_b, lat_c, lon_c)

                            delta_heading = (heading_bc - heading_ab + 180) % 360 - 180

                            if turn_direction * delta_heading < 0 and abs(delta_heading) > 10:
                                is_circling = False
                                break

                            total_heading_change += abs(delta_heading)
                        else:
                            is_circling = total_heading_change >= min_total_heading_change

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

                    if distance_traveled < distance_threshold and is_circling:
                        thermals_data.append(segment)
                    else:
                        sustained_lift_data.append(segment)

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

            # INLINED CIRCLING CHECK LOGIC
            is_circling = False
            if len(segment_points) >= 3:
                total_heading_change = 0
                lat_p1, lon_p1 = segment_points[0][0], segment_points[0][1]
                lat_p2, lon_p2 = segment_points[1][0], segment_points[1][1]
                lat_p3, lon_p3 = segment_points[2][0], segment_points[2][1]

                heading_1 = get_heading(lat_p1, lon_p1, lat_p2, lon_p2)
                heading_2 = get_heading(lat_p2, lon_p2, lat_p3, lon_p3)
                initial_delta_heading = (heading_2 - heading_1 + 180) % 360 - 180
                turn_direction = 1 if initial_delta_heading >= 0 else -1
                total_heading_change += abs(initial_delta_heading)

                for j in range(2, len(segment_points) - 1):
                    lat_a, lon_a = segment_points[j - 2][0], segment_points[j - 2][1]
                    lat_b, lon_b = segment_points[j - 1][0], segment_points[j - 1][1]
                    lat_c, lon_c = segment_points[j][0], segment_points[j][1]

                    heading_ab = get_heading(lat_a, lon_a, lat_b, lon_b)
                    heading_bc = get_heading(lat_b, lon_b, lat_c, lon_c)

                    delta_heading = (heading_bc - heading_ab + 180) % 360 - 180

                    if turn_direction * delta_heading < 0 and abs(delta_heading) > 10:
                        is_circling = False
                        break

                    total_heading_change += abs(delta_heading)
                else:
                    is_circling = total_heading_change >= min_total_heading_change

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

            if distance_traveled < distance_threshold and is_circling:
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


def main():
    """
    Main function to analyze multiple IGC files and plot thermal distributions.
    """
    # --- Input folder to analyze ---
    folder_path = "./igc"

    igc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.igc')]

    if not igc_files:
        print(f"No IGC files found in the folder: {folder_path}. Please check the path and try again.")
        return

    all_thermal_strengths = []
    all_thermal_distances = []
    all_sustained_lift_distances = []
    total_flight_duration_seconds = 0
    total_flight_area_sq_m = 0

    print("Starting multi-file thermal analysis...")

    for filename in igc_files:
        print(f"Processing file: {filename}")
        thermals, sustained_lift_segments, duration_s, flight_path_coords = find_thermals_and_sustained_lift(filename)
        total_flight_duration_seconds += duration_s

        if flight_path_coords and len(flight_path_coords) >= 3:
            points = np.array(flight_path_coords)
            try:
                hull = ConvexHull(points)
                area_in_degrees = hull.area

                avg_lat_rad = np.mean(points[:, 0]) * (math.pi / 180)
                meters_per_lat_deg = 111132
                meters_per_lon_deg = 111320 * math.cos(avg_lat_rad)
                area_sq_m = area_in_degrees * meters_per_lat_deg * meters_per_lon_deg
                total_flight_area_sq_m += area_sq_m

            except Exception as e:
                print(f"Could not calculate convex hull area for {filename}: {e}")

        for thermal in thermals:
            if thermal['climb_rate'] > 0:
                all_thermal_strengths.append(thermal['climb_rate'])

        if len(thermals) > 1:
            for i in range(1, len(thermals)):
                location1 = thermals[i - 1]['end_location']
                location2 = thermals[i]['start_location']
                timestamp1 = thermals[i - 1]['end_timestamp']
                timestamp2 = thermals[i]['start_timestamp']

                distance = haversine_distance(
                    location1[0], location1[1],
                    location2[0], location2[1]
                ) / 1000

                time_gap_seconds = timestamp2 - timestamp1

                if distance > max_merge_distance_km and \
                        distance <= max_thermal_distance_km and \
                        time_gap_seconds <= max_gliding_time_min * 60:
                    all_thermal_distances.append(distance)

        for i in range(1, len(sustained_lift_segments)):
            start_lat1, start_lon1 = sustained_lift_segments[i - 1]['start_location']
            start_lat2, start_lon2 = sustained_lift_segments[i]['start_location']
            distance = haversine_distance(start_lat1, start_lon1, start_lat2, start_lon2)
            all_sustained_lift_distances.append(distance / 1000)

    filtered_thermal_distances = [d for d in all_thermal_distances if d <= max_thermal_distance_km]

    print("\n--- Summary of all analyzed IGC files ---")
    total_thermals = len(all_thermal_strengths)
    print(f"Total thermals identified across all files: {total_thermals}")

    if not all_thermal_strengths:
        print("No thermals were identified in any of the provided files. Cannot perform analysis.")
    else:
        print("\n--- Thermal Strength Distribution ---")
        average_strength = np.mean(all_thermal_strengths)
        print(f"Average Strength: {average_strength:.2f} m/s")
        print(f"Median Strength: {np.median(all_thermal_strengths):.2f} m/s")
        print(f"Standard Deviation: {np.std(all_thermal_strengths):.2f} m/s")

        exponential_lambda = 1 / average_strength if average_strength > 0 else 0
        print(f"Calculated Exponential lambda (λ) parameter: {exponential_lambda:.4f}")

        print(
            f"\n--- Distance Between Thermals Distribution (Filtered with merge dist {max_merge_distance_km}km, max dist {max_thermal_distance_km}km, max time {max_gliding_time_min}min) ---")
        if filtered_thermal_distances:
            average_distance_km = np.mean(filtered_thermal_distances)
            median_distance_km = np.median(filtered_thermal_distances)
            print(f"Average Distance: {average_distance_km:.2f} km")
            print(f"Median Distance: {median_distance_km:.2f} km")
            print(f"Standard Deviation: {np.std(filtered_thermal_distances):.2f} km")

            thermal_rate_per_km = 1 / average_distance_km if average_distance_km > 0 else 0
            print(f"Calculated Linear Thermal Density: {thermal_rate_per_km:.2f} thermals/km")
        else:
            print("Not enough thermals to calculate distances after filtering.")

        print("\n--- Thermal Spatial Density Analysis ---")
        total_flight_area_sq_km = total_flight_area_sq_m / 1_000_000
        if total_flight_area_sq_km > 0 and total_thermals > 0:
            spatial_poisson_lambda = total_thermals / total_flight_area_sq_km
            print(f"Total flight area (estimated): {total_flight_area_sq_km:.2f} km^2")
            print(f"Calculated Spatial Poisson lambda (λ) parameter: {spatial_poisson_lambda:.5f} thermals/km^2")
        else:
            print("Not enough flight data to calculate thermal spatial density.")

    print("\n--- Sustained Lift Segment Analysis (Non-Circling) ---")
    total_sustained_segments = len(all_sustained_lift_distances) + 1 if len(all_sustained_lift_distances) > 0 else 0
    print(f"Total sustained lift segments identified: {total_sustained_segments}")
    if all_sustained_lift_distances:
        average_sustained_distance_km = np.mean(all_sustained_lift_distances)
        print(f"Average Distance Between Segments: {average_sustained_distance_km:.2f} km")
    else:
        print("Not enough sustained lift segments to calculate distances.")

    # --- Plot the distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(all_thermal_strengths, bins=50, density=True, color='skyblue', edgecolor='black')
    axes[0].set_title('Probability Distribution of Thermal Strength')
    axes[0].set_xlabel('Thermal Strength (Average Climb Rate in m/s)')
    axes[0].set_ylabel('Probability Density')
    axes[0].grid(axis='y', alpha=0.75)

    if filtered_thermal_distances:
        axes[1].hist(filtered_thermal_distances, bins=15, density=True, color='lightgreen', edgecolor='black')
        axes[1].set_title(f'Probability Distribution of Thermal Distance (filtered)')
        axes[1].set_xlabel('Distance Between Thermals (km)')
        axes[1].set_ylabel('Probability Density')
        axes[1].grid(axis='y', alpha=0.75)
    else:
        axes[1].set_title('Not enough thermals to plot distances after filtering')

    plt.suptitle('Thermal Distribution Analysis Across All Flights')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
