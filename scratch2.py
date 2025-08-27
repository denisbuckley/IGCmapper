import csv
import math

#
# This script reads thermal data from a CSV file, filters the points that
# lie within a defined cone along a flight path, and exports the filtered
# data to a new CSV file. The start and end points of the flight path are
# now chosen by the user from a list of waypoints read from a .cup file.
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
# Define the cone angle and tolerance for the distance check.
CONE_ANGLE_DEG = 30  # The cone angle in degrees (20 degrees either side of the path)
TOLERANCE_KM = 5
# File names for input and output
WAYPOINT_FILE = 'gcwa extended.cup'
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

def parse_coords(coord_str):
    """
    Parses a coordinate string from the .cup file (e.g., '3222.447S') and converts it to
    a signed decimal degree float.
    Args:
        coord_str: The coordinate string.
    Returns:
        The coordinate in decimal degrees.
    """
    direction = coord_str[-1].upper()
    value = float(coord_str[:-1])
    degrees = math.floor(value / 100)
    minutes = value % 100
    decimal_degrees = degrees + minutes / 60
    if direction in ['S', 'W']:
        return -decimal_degrees
    return decimal_degrees


def read_waypoints_from_cup(file_path):
    """
    Reads waypoints from a .cup file and returns a list of dictionaries.
    Args:
        file_path: The path to the .cup file.
    Returns:
        A list of dictionaries, where each dictionary contains a waypoint's
        'name', 'code', 'lat', and 'lon'. Returns an empty list on error.
    """
    waypoints = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Find the header row
            header_row = next(reader)
            while header_row[0].strip().upper() != 'TITLE':
                header_row = next(reader)

            # Process the remaining rows
            for row in reader:
                # The columns are defined as Title,Code,Country,Latitude,Longitude,Elevation
                if len(row) > 4:
                    waypoints.append({
                        'name': row[0].strip('"'),
                        'code': row[1].strip('"'),
                        'lat': parse_coords(row[3].strip('"')),
                        'lon': parse_coords(row[4].strip('"'))
                    })
    except FileNotFoundError:
        print(f"Error: The waypoint file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading the waypoint file: {e}")
        return []
    return waypoints


def main():
    """
    Main function to read data, filter, and write to a new CSV file.
    It now prompts the user to select waypoints for the flight path.
    """
    # 1. Read waypoints from the .cup file
    print(f"Reading waypoints from '{WAYPOINT_FILE}'...")
    waypoints = read_waypoints_from_cup(WAYPOINT_FILE)
    if not waypoints:
        return  # Exit if waypoints couldn't be loaded

    # 2. Present waypoints and get user input for start and end points
    print("\nAvailable Waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"[{i+1}] {wp['name']} ({wp['code']})")

    def get_user_waypoint(prompt):
        while True:
            try:
                choice = int(input(f"\nEnter the number for the {prompt} waypoint: "))
                if 1 <= choice <= len(waypoints):
                    return waypoints[choice - 1]
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(waypoints)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    start_waypoint = get_user_waypoint("start")
    end_waypoint = get_user_waypoint("end")

    start_lat, start_lon = start_waypoint['lat'], start_waypoint['lon']
    end_lat, end_lon = end_waypoint['lat'], end_waypoint['lon']

    print(f"\nFiltering thermals for a flight path from '{start_waypoint['name']}' to '{end_waypoint['name']}'...")
    print(f"Start Coords: ({start_lat}, {start_lon})")
    print(f"End Coords: ({end_lat}, {end_lon})")

    # 3. Read thermal data and apply the filter
    print(f"\nReading thermal data from '{INPUT_FILE}'...")
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
                        lat, lon, strength = float(row[0]), float(row[1]), float(row[2])

                        # Apply the filter with the user-selected waypoints
                        if is_within_cone(lat, lon, start_lat, start_lon, end_lat, end_lon, CONE_ANGLE_DEG):
                            writer.writerow(row)  # Write the row to the new file if it passes
                            rows_filtered += 1

                    except (ValueError, IndexError) as e:
                        # Skip malformed rows but notify the user
                        print(f"Warning: Skipping malformed row '{row}'. Error: {e}")

        print(f"\nFiltering complete. Wrote {rows_filtered} rows to '{OUTPUT_FILE}'.")

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_FILE}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

