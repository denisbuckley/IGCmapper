import csv
import math

#
# This script reads thermal data from a CSV file, filters the points that
# lie within a defined cone along a flight path, and exports the filtered
# data to a new CSV file.
#
# The code's logic works in two stages:
#
# 1. The "On the Line" Check: It first calculates the total distance of the flight
#    path. Then, for each thermal, it checks if the distance from the start to the
#    thermal plus the distance from the thermal to the end is approximately equal
#    to the total path distance. If this is true, the thermal must lie very close
#    to the straight line defined by the flight path.
#
# 2. The "In the Cone" Check: It then calculates the bearing (direction) from the
#    start point to the end point. It compares this with the bearing from the
#    start point to the thermal. If the difference between these two bearings is
#    less than the CONE_ANGLE_DEG, the thermal is considered to be within the
#    desired sector.
#
# By combining these two checks, the script efficiently and accurately filters for thermals
# that are both on the flight path and within the specified cone.
#
# Required libraries: none beyond standard Python.


# --- Configuration ---
# Define the flight path start and end points in (latitude, longitude)
# Coordinates are in decimal degrees.

START_LAT, START_LON = -31.66, 117.24
END_LAT, END_LON = -30.596, 116.772
CONE_ANGLE_DEG = 20  # The cone angle in degrees (20 degrees either side of the path)

# File names for input and output
INPUT_FILE = 'consolidated_thermal_coords.csv'
OUTPUT_FILE = 'filtered_thermals.csv'


# --- Geospatial Calculation Functions ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two points on the Earth using the Haversine formula.
    Args:
        lat1, lon1: Coordinates of the first point in decimal degrees.
        lat2, lon2: Coordinates of the second point in decimal degrees.
    Returns:
        The distance in kilometers.
    """
    R = 6371  # Radius of Earth in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the initial bearing between two points.
    Args:
        lat1, lon1: Coordinates of the first point in decimal degrees.
        lat2, lon2: Coordinates of the second point in decimal degrees.
    Returns:
        The bearing in degrees (0-360).
    """
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    return (bearing_deg + 360) % 360


def is_within_cone(thermal_lat, thermal_lon, start_lat, start_lon, end_lat, end_lon, cone_angle_deg):
    """
    Determines if a thermal point is within the defined cone.
    Args:
        thermal_lat, thermal_lon: Coordinates of the thermal.
        start_lat, start_lon: Coordinates of the start of the flight path.
        end_lat, end_lon: Coordinates of the end of the flight path.
        cone_angle_deg: The half-angle of the cone in degrees.
    Returns:
        True if the thermal is in the cone, False otherwise.
    """
    # Use a small tolerance for floating point comparisons
    TOLERANCE_KM = 0.5

    # Calculate the total distance of the flight path.
    total_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)

    # Calculate the distance from the start point to the thermal and from the thermal to the end point.
    dist_to_thermal = haversine_distance(start_lat, start_lon, thermal_lat, thermal_lon)
    dist_thermal_to_end = haversine_distance(thermal_lat, thermal_lon, end_lat, end_lon)

    # Check if the thermal is approximately between the start and end points.
    # The sum of the two smaller distances should be close to the total distance.
    if not math.isclose(dist_to_thermal + dist_thermal_to_end, total_distance, abs_tol=TOLERANCE_KM):
        return False

    # Calculate the bearing from the start point to the end point (path bearing)
    path_bearing = calculate_bearing(start_lat, start_lon, end_lat, end_lon)

    # Calculate the bearing from the start point to the thermal point
    thermal_bearing = calculate_bearing(start_lat, start_lon, thermal_lat, thermal_lon)

    # Calculate the difference in bearings and handle the 360-degree wrap-around
    bearing_diff = abs(path_bearing - thermal_bearing)
    if bearing_diff > 180:
        bearing_diff = 360 - bearing_diff

    # Check if the bearing difference is within the cone angle
    return bearing_diff <= cone_angle_deg


# --- Main Logic ---

def main():
    """
    Main function to read data, filter, and write to a new CSV file.
    """
    print(f"Reading data from '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Read the header row

            with open(OUTPUT_FILE, mode='w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)  # Write the header to the new file

                rows_filtered = 0
                for row in reader:
                    try:
                        # Parse the data from the row
                        lat, lon, strength = float(row[0]), float(row[1]), int(row[2])

                        # Apply the filter
                        if is_within_cone(lat, lon, START_LAT, START_LON, END_LAT, END_LON, CONE_ANGLE_DEG):
                            writer.writerow(row)  # Write the row to the new file if it passes
                            rows_filtered += 1

                    except (ValueError, IndexError) as e:
                        # Skip malformed rows but notify the user
                        print(f"Warning: Skipping malformed row '{row}'. Error: {e}")

        print(f"Filtering complete. Wrote {rows_filtered} rows to '{OUTPUT_FILE}'.")

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_FILE}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
