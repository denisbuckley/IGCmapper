#
# This script manually reads and parses an IGC file, plotting the flight path
# and marking potential thermals and circling areas. It does not require a
# third-party IGC parsing library.
#
# Required library: matplotlib
# Install with: pip install matplotlib
#

import matplotlib.pyplot as plt
import os
import math


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


def plot_igc_with_thermals(filepath):
    """
    Reads an IGC file, manually parses the flight data, identifies potential
    thermals and circling areas, and plots the flight path with them marked.

    Args:
        filepath (str): The path to the IGC file.
    """
    try:
        latitudes = []
        longitudes = []
        altitudes = []
        times = []
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

                        # Extract time (e.g., '120101')
                        times.append(line[1:7])

                    except (ValueError, IndexError) as e:
                        print(f"Warning: Failed to parse B-record line: '{line.strip()}' due to error: {e}")
                        continue

        if not latitudes:
            print("Error: The IGC file was read, but no B-records (GPS fixes) were found.")
            print("Please ensure the IGC file is not corrupted and contains valid flight data.")
            return

        # --- 3. Identify potential thermals (simple altitude gain) ---
        thermals_lat = []
        thermals_lon = []
        altitude_change_threshold = 20  # meters, adjust as needed

        print("Analyzing flight data for thermals...")
        for i in range(1, len(altitudes)):
            altitude_gain = altitudes[i] - altitudes[i - 1]
            if altitude_gain > altitude_change_threshold:
                thermals_lat.append(latitudes[i])
                thermals_lon.append(longitudes[i])

        print(f"Found {len(thermals_lat)} potential thermals based on altitude gain.")

        # --- 4. Identify circling/stationary periods (advanced heuristic) ---
        # This heuristic looks for periods of sustained altitude gain while the glider
        # is also staying within a small geographical area, indicating circling.
        circling_lat = []
        circling_lon = []
        time_window = 10  # Check against a point 10 seconds ago
        distance_threshold = 100  # meters, max distance traveled in the time window

        print("Analyzing flight data for circling periods...")
        for i in range(time_window, len(altitudes)):
            # Check for sustained altitude gain over the time window
            altitude_diff = altitudes[i] - altitudes[i - time_window]

            # Check for a small geographical distance traveled over the time window
            distance_traveled = haversine_distance(
                latitudes[i], longitudes[i],
                latitudes[i - time_window], longitudes[i - time_window]
            )

            if altitude_diff > altitude_change_threshold and distance_traveled < distance_threshold:
                circling_lat.append(latitudes[i])
                circling_lon.append(longitudes[i])

        print(f"Found {len(circling_lat)} potential circling points.")

        # --- 5. Plot the flight path, thermals, and circling points ---
        plt.figure(figsize=(10, 8))

        plt.plot(longitudes, latitudes, color='blue', label='Flight Path', zorder=1)

        plt.scatter(thermals_lon, thermals_lat, c='red', marker='x', s=100, label='Altitude Gain', zorder=2)

        plt.scatter(circling_lon, circling_lat, c='green', marker='o', s=50, label='Circling/Stationary', zorder=3)

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
    if os.path.exists(igc_file):
        plot_igc_with_thermals(igc_file)
    else:
        print(f"The file '{igc_file}' does not exist.")
