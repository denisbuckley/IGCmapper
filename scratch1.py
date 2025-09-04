# This script analyzes multiple IGC files to determine the probability distributions
# of thermal distance and strength. It also calculates the exponential lambda,
# and a spatial Poisson lambda (thermal density) in thermals per km^2.
# It has been updated to also analyze sustained segments of lift that are not thermals.
#
# Required libraries: matplotlib, numpy, scipy
# Install with: pip install matplotlib numpy scipy
#


import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.spatial import ConvexHull

# --- User-configurable variables ---
# Heuristic parameters for identifying a thermal (a sustained circling period).
time_window = 30  # seconds to check for sustained climb and confined area
distance_threshold = 300  # meters, max distance traveled in the time window
altitude_change_threshold = 30  # meters
# New parameter to merge thermal segments separated by short gaps.
max_gap_seconds = 20  # seconds, maximum time gap to consider two segments part of the same thermal
# New parameter to filter out large distances that skew the distribution.
max_thermal_distance_km = 20  # kilometers, maximum distance to consider between thermals
# New parameter to group closely spaced thermals into a single event for distance calculation.
max_merge_distance_km = 3
# kilometers, maximum distance to consider two thermals as a single event
# NEW: Filter out distances based on time. Prevents linking thermals separated by long breaks.
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

                        latitudes.append(lat)
                        longitudes.append(lon)
                        altitudes.append(alt)
                        timestamps_seconds.append(time_s)

                    except (ValueError, IndexError):
                        continue

        if not latitudes:
            print(f"Warning: No valid GPS points found in {filepath}. Skipping.")
            return [], [], 0, []

        # Find sustained lift points (either circling or linear)
        sustained_lift_points_indices = []
        for i in range(time_window, len(altitudes)):
            altitude_diff = altitudes[i] - altitudes[i - time_window]
            distance_traveled = haversine_distance(
                latitudes[i], longitudes[i],
                latitudes[i - time_window], longitudes[i - time_window]
            )
            if altitude_diff > altitude_change_threshold:
                sustained_lift_points_indices.append(i)

        # Group successive sustained lift points into distinct segments,
        # with tolerance for short gaps.
        thermals_data = []
        sustained_lift_data = []

        if not sustained_lift_points_indices:
            flight_duration = timestamps_seconds[-1] - timestamps_seconds[0] if len(timestamps_seconds) > 1 else 0
            flight_path = list(zip(latitudes, longitudes))
            return [], [], flight_duration, flight_path

        # Initialize the first segment
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

        # Process the last segment
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


def main():
    """
    Main function to analyze multiple IGC files and plot thermal distributions.
    """
    # --- Input folder to analyze ---
    # Put your IGC files in a folder named 'igc' in the same directory as this script.
    folder_path = "./igc"

    # Filter for IGC files
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
                # Calculate area of the 2D polygon defined by the convex hull vertices
                area_in_degrees = hull.area

                # Approximate conversion from degrees^2 to m^2. This is an approximation
                # and assumes a relatively small, flat area. The conversion factor
                # changes with latitude, so we use a general approximation.
                # 1 degree lat approx 111,132 m, 1 degree lon approx 111,320 * cos(lat) m
                # We'll use a rough average value for simplicity.
                avg_lat_rad = np.mean(points[:, 0]) * (math.pi / 180)
                meters_per_lat_deg = 111132
                meters_per_lon_deg = 111320 * math.cos(avg_lat_rad)
                area_sq_m = area_in_degrees * meters_per_lat_deg * meters_per_lon_deg
                total_flight_area_sq_m += area_sq_m

            except Exception as e:
                print(f"Could not calculate convex hull area for {filename}: {e}")

        # Collect thermal strengths
        for thermal in thermals:
            if thermal['climb_rate'] > 0:
                all_thermal_strengths.append(thermal['climb_rate'])

        # --- NEW thermal distance calculation logic for distance calculation ---
        # This calculates distances between consecutive thermals and filters out
        # unrealistically large distances that skew the statistical analysis.
        if len(thermals) > 1:
            for i in range(1, len(thermals)):
                # Get the end location of the previous thermal
                location1 = thermals[i - 1]['end_location']
                # Get the start location of the current thermal
                location2 = thermals[i]['start_location']
                # Get the end timestamp of the previous thermal
                timestamp1 = thermals[i - 1]['end_timestamp']
                # Get the start timestamp of the current thermal
                timestamp2 = thermals[i]['start_timestamp']

                distance = haversine_distance(
                    location1[0], location1[1],
                    location2[0], location2[1]
                ) / 1000  # Distance in km

                time_gap_seconds = timestamp2 - timestamp1

                # Filter out small distances between thermals that are effectively
                # part of the same thermal event, as defined by max_merge_distance_km.
                # Also, filter out distances where the time gap is too long,
                # indicating a break in the flight.
                if distance > max_merge_distance_km and \
                        distance <= max_thermal_distance_km and \
                        time_gap_seconds <= max_gliding_time_min * 60:
                    all_thermal_distances.append(distance)
        # -----------------------------------------------------------------------

        # Calculate and collect distances between successive sustained lift segments for this flight
        for i in range(1, len(sustained_lift_segments)):
            start_lat1, start_lon1 = sustained_lift_segments[i - 1]['start_location']
            start_lat2, start_lon2 = sustained_lift_segments[i]['start_location']
            distance = haversine_distance(start_lat1, start_lon1, start_lat2, start_lon2)
            all_sustained_lift_distances.append(distance / 1000)  # Store in kilometers

    # The filtering is now handled within the loop, so this line is less critical but still useful
    filtered_thermal_distances = [d for d in all_thermal_distances if d <= max_thermal_distance_km]

    print("\n--- Summary of all analyzed IGC files ---")
    total_thermals = len(all_thermal_strengths)
    print(f"Total thermals identified across all files: {total_thermals}")
    print(
        f"Thermals included in distance analysis (after filtering): {len(filtered_thermal_distances)} [{total_thermals}]")

    if not all_thermal_strengths:
        print("No thermals were identified in any of the provided files. Cannot perform analysis.")
    else:
        # --- Print Summary Statistics for Thermals ---
        print("\n--- Thermal Strength Distribution ---")
        average_strength = np.mean(all_thermal_strengths)
        print(f"Average Strength: {average_strength:.2f} m/s")
        print(f"Median Strength: {np.median(all_thermal_strengths):.2f} m/s")
        print(f"Standard Deviation: {np.std(all_thermal_strengths):.2f} m/s")

        # Calculate lambda for the exponential distribution
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

            # Calculate the probability of encountering a thermal per kilometer (linear density)
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

    # Plot 1: Thermal Strength Distribution
    axes[0].hist(all_thermal_strengths, bins=50, density=True, color='skyblue', edgecolor='black')
    axes[0].set_title('Probability Distribution of Thermal Strength')
    axes[0].set_xlabel('Thermal Strength (Average Climb Rate in m/s)')
    axes[0].set_ylabel('Probability Density')
    axes[0].grid(axis='y', alpha=0.75)

    # Plot 2: Distance Between Thermals Distribution
    if filtered_thermal_distances:
        axes[1].hist(filtered_thermal_distances, bins=15, density=True, color='lightgreen', edgecolor='black')
        axes[1].set_title(
            f'Probability Distribution of Thermal Distance ({len(filtered_thermal_distances)} [{total_thermals}] thermals)')
        axes[1].set_xlabel('Distance Between Thermals (km)')
        axes[1].set_ylabel('Probability Density')
        axes[1].grid(axis='y', alpha=0.75)
    else:
        axes[1].set_title('Not enough thermals to plot distances after filtering')

    plt.suptitle('Thermal Distribution Analysis Across All Flights')

    # Add text box for the parameter values at the bottom of the figure
    param_text = (
        f"Parameters Used:\n"
        f"Time Window: {time_window}s, Distance Threshold: {distance_threshold}m, Altitude Gain: {altitude_change_threshold}m\n"
        f"Merge Gap: {max_gap_seconds}s, Max Thermal Distance: {max_thermal_distance_km}km, Merge Distance: {max_merge_distance_km}km\n"
        f"Max Gliding Time: {max_gliding_time_min}min"
    )
    plt.figtext(0.5, 0.01, param_text, ha="center", fontsize=10, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
