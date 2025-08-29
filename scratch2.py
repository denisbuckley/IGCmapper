import csv
import math

# Import all functions from the new holder file
from thermal_filter_functions_holder import *

#
# This script reads thermal data from a CSV file, filters the points that
# lie within a defined cone along a flight path, and exports the filtered
# data to a new CSV, CUP, and KML file. The start and end points of the flight path are
# now chosen by the user from a list of waypoints read from a .cup file.
# The user is now prompted to enter a value only for the cone angle.
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


# The script calls the following functions:
#
# haversine_distance(): Calculates the distance between two geographical points.
# calculate_bearing(): Determines the bearing (direction) from one point to another.
# is_within_cone(): The core logic for filtering thermals, using the haversine_distance and calculate_bearing functions.
# parse_coords(): Converts coordinates from the .cup file's format to standard decimal degrees.
# convert_to_cup_coord(): Converts decimal degrees back to the .cup file's coordinate format.
# read_waypoints_from_cup(): Reads the waypoints from the gcwa extended.cup file.
# get_float_input(): Prompts the user for a numerical input with error handling.
# write_cup_file(): Creates a new .cup file from the filtered data.
# write_kml_file(): Creates a new .kml file from the filtered data.


# Required libraries: none beyond standard Python.


# --- Configuration ---
# File names for input and output
WAYPOINT_FILE = 'gcwa extended.cup'
INPUT_FILE = 'consolidated_thermal_coords.csv'
OUTPUT_CSV_FILE = 'filtered_thermals.csv'
OUTPUT_CUP_FILE = 'filtered_thermals.cup'
OUTPUT_KML_FILE = 'filtered_thermals.kml'


def main():
    """
    Main function to read data, filter, and write to new CSV and CUP files.
    It now prompts the user to select waypoints for the flight path and
    enter filter parameters.
    """
    # 1. Read waypoints from the .cup file
    print(f"Reading waypoints from '{WAYPOINT_FILE}'...")
    waypoints = read_waypoints_from_cup(WAYPOINT_FILE)
    if not waypoints:
        return  # Exit if waypoints couldn't be loaded

    # 2. Present waypoints and get user input for start and end points
    print("\nAvailable Waypoints:")
    for i, wp in enumerate(waypoints):
        print(f"[{i + 1}] {wp['name']} ({wp['code']})")

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

    # 3. Prompt user for cone angle and calculate the tolerance
    cone_angle = get_float_input("Enter the cone angle in degrees (e.g., 20): ", 30)

    # The tolerance is dynamically calculated based on the flight path distance and cone angle.
    # This models a widening search corridor where the maximum lateral deviation occurs at the
    # midpoint of the path. The formula is: tolerance = (distance_of_flight_path / 2) * tan(cone_angle / 2).
    total_distance = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    tolerance = (total_distance / 2) * math.tan(math.radians(cone_angle / 2))

    print(f"\nFiltering thermals for a flight path from '{start_waypoint['name']}' to '{end_waypoint['name']}'...")
    print(f"Using Cone Angle: {cone_angle} degrees, Calculated Tolerance: {tolerance:.2f} km")
    print(f"Start Coords: ({start_lat}, {start_lon})")
    print(f"End Coords: ({end_lat}, {end_lon})")

    # 4. Read thermal data and apply the filter, writing to the new CSV file
    print(f"\nReading thermal data from '{INPUT_FILE}'...")
    try:
        filtered_rows = []
        with open(INPUT_FILE, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Read the header row

            with open(OUTPUT_CSV_FILE, mode='w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(header)  # Write the header to the new CSV file

                for row in reader:
                    try:
                        # Parse the data from the row
                        lat, lon, strength = float(row[0]), float(row[1]), float(row[2])

                        # Apply the filter with the user-selected waypoints and parameters
                        if is_within_cone(lat, lon, start_lat, start_lon, end_lat, end_lon, cone_angle, tolerance):
                            writer.writerow(row)  # Write the row to the new CSV file
                            filtered_rows.append(row)

                    except (ValueError, IndexError) as e:
                        # Skip malformed rows but notify the user
                        print(f"Warning: Skipping malformed row '{row}'. Error: {e}")

        print(f"\nFiltering complete. Wrote {len(filtered_rows)} rows to '{OUTPUT_CSV_FILE}'.")

        # 5. Write the CUP and KML files from the filtered CSV data
        if len(filtered_rows) > 0:
            write_cup_file(OUTPUT_CSV_FILE, OUTPUT_CUP_FILE)
            write_kml_file(OUTPUT_CSV_FILE, OUTPUT_KML_FILE)
        else:
            print("No thermals were filtered. Skipping file generation.")

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_FILE}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
