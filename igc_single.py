# This script manually reads, parses, and analyzes an IGC flight file to identify
# and plot potential thermals and significant climbs. It relies solely on built-in
# Python functionality and the `matplotlib` library for plotting, without
# requiring any third-party IGC parsing libraries.
#

# Functions Called:
# - haversine_distance: Calculates the distance between two geographic points.
# - igc_to_decimal_degrees: Converts IGC format coordinates to decimal degrees.
# - time_to_seconds: Converts a time string to total seconds from midnight.
# - plot_igc_with_thermals: The main function that orchestrates the parsing,
#   analysis, and plotting.
#
# Required library: matplotlib
# Install with: pip install matplotlib
#

import matplotlib.pyplot as plt
import os
import math

# --- User-configurable variables ---
# Threshold for a simple altitude gain (red 'x' markers)
altitude_change_threshold = 20  # meters

# Heuristic parameters for identifying circling/stationary flight (green 'o' markers)
# This heuristic assumes that a thermal is being worked if the pilot gains
# altitude while staying within a confined area, defined by the variables below.
time_window = 10  # seconds to check for sustained climb and confined area
distance_threshold = 100  # meters, max distance traveled in the time window

# Threshold for identifying a significant, continuous climb (yellow '^' markers)
significant_climb_threshold = 200  # meters


def igc_to_decimal_degrees(igc_coord):
    """
    Converts a coordinate from IGC format (DDMMmmmN/S/E/W) to decimal degrees.
    e.g., "5101234N" becomes 51.02056666...
    """
    # The last character is the direction (N/S/E/W)
    direction = igc_coord[-1]

    # Degrees are the first two digits for latitude, or three for longitude
    if direction in 'NS':
        degrees = float(igc_coord[:2])
        minutes = float(igc_coord[2:-1]) / 1000.0
    else:  # For Longitude
        degrees = float(igc_coord[:3])
        minutes = float(igc_coord[3:-1]) / 1000.0

    decimal_degrees = degrees + (minutes / 60.0)

    # Apply negative sign for South and West
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
    e.g., '120101' becomes 12 * 3600 + 1 * 60 + 1 = 43261 seconds.
    """
    try:
        hours = int(time_str[0:2])
        minutes = int(time_str[2:4])
        seconds = int(time_str[4:6])
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, IndexError):
        return None


def plot_igc_with_thermals(filepath):
    """
    Reads an IGC file, manually parses the flight data, identifies potential
    thermals and circling areas, and plots the flight path with them marked.
    Also, identifies significant climbs and their corresponding horizontal distance.

    Args:
        filepath (str): The path to the IGC file.
    """
    try:
        latitudes = []
        longitudes = []
        altitudes = []
        timestamps_seconds = []

        pilot_name = "Unknown Pilot"

        with open(filepath, 'r') as file:
            for line in file:
                record_type = line[0]

                # Extract pilot name from header
                if record_type == 'H' and line[3:6] == 'PLT':
                    pilot_name = line[6:].strip()

                # Parse B-records for flight data
                if record_type == 'B' and len(line) >= 35:
                    try:
                        # Extract and convert time to seconds from midnight
                        time_str = line[1:7]
                        time_s = time_to_seconds(time_str)

                        # Extract and convert latitude (e.g., '5101234N')
                        lat_igc = line[7:15]
                        lat = igc_to_decimal_degrees(lat_igc)

                        # Extract and convert longitude (e.g., '00700400W')
                        lon_igc = line[15:24]
                        lon = igc_to_decimal_degrees(lon_igc)

                        # Extract barometric altitude (e.g., '01423')
                        alt = int(line[25:30])

                        latitudes.append(lat)
                        longitudes.append(lon)
                        altitudes.append(alt)
                        timestamps_seconds.append(time_s)

                    except (ValueError, IndexError) as e:
                        print(f"Warning: Failed to parse B-record line: '{line.strip()}' due to error: {e}")
                        continue

        if not latitudes:
            print("Error: The IGC file was read, but no B-records (GPS fixes) were found.")
            print("Please ensure the IGC file is not corrupted and contains valid flight data.")
            return

        print(f"Successfully parsed {len(latitudes)} GPS points from the IGC file.")

        # --- 1. Identify circling/stationary periods (advanced heuristic) ---
        circling_points_indices = []

        print("Analyzing flight data for circling periods...")
        for i in range(time_window, len(altitudes)):
            altitude_diff = altitudes[i] - altitudes[i - time_window]
            distance_traveled = haversine_distance(
                latitudes[i], longitudes[i],
                latitudes[i - time_window], longitudes[i - time_window]
            )

            if altitude_diff > altitude_change_threshold and distance_traveled < distance_threshold:
                circling_points_indices.append(i)

        print(f"Found {len(circling_points_indices)} potential circling points.")

        # --- 2. Group successive circling periods into single thermals ---
        thermals_data = []
        if circling_points_indices:
            current_thermal = {
                'start_index': circling_points_indices[0],
                'end_index': circling_points_indices[0],
                'altitude_gain': altitudes[circling_points_indices[0]] - altitudes[
                    circling_points_indices[0] - time_window]
            }

            for i in range(1, len(circling_points_indices)):
                current_point_index = circling_points_indices[i]
                previous_point_index = circling_points_indices[i - 1]

                # Check if the current point is successive to the previous one
                if current_point_index == previous_point_index + 1:
                    current_thermal['end_index'] = current_point_index
                    current_thermal['altitude_gain'] += (
                                altitudes[current_point_index] - altitudes[current_point_index - time_window])
                else:
                    # New thermal starts
                    thermals_data.append(current_thermal)
                    current_thermal = {
                        'start_index': current_point_index,
                        'end_index': current_point_index,
                        'altitude_gain': altitudes[current_point_index] - altitudes[current_point_index - time_window]
                    }
            thermals_data.append(current_thermal)

        # --- 3. Identify significant continuous climbs and their distance and strength ---
        significant_climb_segments = []

        print("Analyzing flight data for significant climbs...")
        in_climb = False
        climb_start_index = 0
        for i in range(1, len(altitudes)):
            # Check for the start of a new climb
            if not in_climb and altitudes[i] > altitudes[i - 1]:
                in_climb = True
                climb_start_index = i - 1

            # Check for the end of a climb
            if in_climb and altitudes[i] <= altitudes[i - 1]:
                climb_end_index = i
                total_altitude_gain = altitudes[climb_end_index - 1] - altitudes[climb_start_index]

                if total_altitude_gain >= significant_climb_threshold:
                    distance = haversine_distance(
                        latitudes[climb_start_index], longitudes[climb_start_index],
                        latitudes[climb_end_index - 1], longitudes[climb_end_index - 1]
                    )

                    # Calculate duration using timestamps for accuracy
                    duration = timestamps_seconds[climb_end_index - 1] - timestamps_seconds[climb_start_index]
                    climb_rate = total_altitude_gain / duration if duration > 0 else 0

                    significant_climb_segments.append({
                        'start_index': climb_start_index,
                        'end_index': climb_end_index,
                        'altitude_gain': total_altitude_gain,
                        'distance': distance,
                        'climb_rate': climb_rate
                    })
                in_climb = False

        # Handle case where climb extends to the end of the file
        if in_climb:
            climb_end_index = len(altitudes)
            total_altitude_gain = altitudes[climb_end_index - 1] - altitudes[climb_start_index]
            if total_altitude_gain >= significant_climb_threshold:
                distance = haversine_distance(
                    latitudes[climb_start_index], longitudes[climb_start_index],
                    latitudes[climb_end_index - 1], longitudes[climb_end_index - 1]
                )
                duration = timestamps_seconds[climb_end_index - 1] - timestamps_seconds[climb_start_index]
                climb_rate = total_altitude_gain / duration if duration > 0 else 0

                significant_climb_segments.append({
                    'start_index': climb_start_index,
                    'end_index': climb_end_index,
                    'altitude_gain': total_altitude_gain,
                    'distance': distance,
                    'climb_rate': climb_rate
                })

        # --- 4. Print the details of all detected climbs and thermals ---
        print("\n--- Thermal (Grouped Circling Periods) Analysis ---")
        if thermals_data:
            print(f"Found {len(thermals_data)} distinct thermal events.")
            for i, thermal in enumerate(thermals_data):
                duration = timestamps_seconds[thermal['end_index']] - timestamps_seconds[thermal['start_index']]
                climb_rate = thermal['altitude_gain'] / duration if duration > 0 else 0
                print(f"  Thermal Event {i + 1}:")
                print(
                    f"    Start Location: Lat {latitudes[thermal['start_index']]:.5f}, Lon {longitudes[thermal['start_index']]:.5f}")
                print(f"    Total Altitude Gain: {thermal['altitude_gain']:.2f} meters")
                print(f"    Duration: {duration} seconds")
                print(f"    Average Climb Rate: {climb_rate:.2f} m/s")
        else:
            print("No thermals (circling periods) found.")

        print("\n--- Significant Climb Analysis ---")
        total_climb_rate = 0
        if significant_climb_segments:
            for i, segment in enumerate(significant_climb_segments):
                print(f"Climb Segment {i + 1}:")
                print(f"  Altitude Gain: {segment['altitude_gain']:.2f} meters")
                print(f"  Horizontal Distance: {segment['distance']:.2f} meters")
                print(f"  Climb Rate (Strength): {segment['climb_rate']:.2f} m/s")
                total_climb_rate += segment['climb_rate']

            avg_climb_rate = total_climb_rate / len(significant_climb_segments)
            print(f"\nAverage Climb Rate (Thermal Strength) for all segments: {avg_climb_rate:.2f} m/s")
        else:
            print("No significant climbs found to analyze.")

        # --- 5. Plot the flight path and significant climbs ---
        plt.figure(figsize=(10, 8))

        plt.plot(longitudes, latitudes, color='blue', label='Flight Path', zorder=1)

        # Plot grouped thermals
        if thermals_data:
            thermal_lons = [longitudes[t['start_index']] for t in thermals_data]
            thermal_lats = [latitudes[t['start_index']] for t in thermals_data]
            plt.scatter(thermal_lons, thermal_lats, c='green', marker='o', s=150, label=f'Thermals (Grouped Circling)',
                        zorder=3)

        significant_climb_lon = [longitudes[seg['end_index']] for seg in significant_climb_segments]
        significant_climb_lat = [latitudes[seg['end_index']] for seg in significant_climb_segments]
        plt.scatter(significant_climb_lon, significant_climb_lat, c='yellow', marker='^', s=150,
                    label=f'Significant Climb (>{significant_climb_threshold}m)', zorder=4)

        # --- 6. Add labels, title, and a legend for clarity ---
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.title(f'IGC Flight Path with Thermal Indications for {pilot_name}')

        plt.legend()
        plt.grid(True)
        plt.show()


    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found. Please check the path and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- User Input Section ---
if __name__ == "__main__":
    igc_file = input("Please enter the path to your IGC file: ")
    if not igc_file.lower().endswith(".igc"):
        igc_file += ".igc"

    if os.path.exists(igc_file):
        plot_igc_with_thermals(igc_file)
    else:
        print(f"The file '{igc_file}' does not exist.")
