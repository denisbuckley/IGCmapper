# thermal_filter_functions_holder.py

import csv
import math

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


def is_within_cone(thermal_lat, thermal_lon, start_lat, start_lon, end_lat, end_lon, cone_angle_deg, tolerance_km):
    """
    Determines if a thermal point is within the defined cone.
    Args:
        thermal_lat, thermal_lon: Coordinates of the thermal.
        start_lat, start_lon: Coordinates of the start of the flight path.
        end_lat, end_lon: Coordinates of the end of the flight path.
        cone_angle_deg: The half-angle of the cone in degrees.
        tolerance_km: The tolerance for the distance check in kilometers.
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
    if not math.isclose(dist_to_thermal + dist_thermal_to_end, total_distance, abs_tol=tolerance_km):
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


# --- File Handling Functions ---

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


def convert_to_cup_coord(decimal_degrees, is_lat=True):
    """
    Converts a decimal degree coordinate to the CUP format (DDMM.MMM).
    Args:
        decimal_degrees: The coordinate in signed decimal degrees.
        is_lat: True for latitude, False for longitude.
    Returns:
        The formatted coordinate string.
    """
    abs_degrees = abs(decimal_degrees)
    degrees = int(abs_degrees)
    minutes = (abs_degrees - degrees) * 60

    if is_lat:
        direction = 'S' if decimal_degrees < 0 else 'N'
    else:
        direction = 'W' if decimal_degrees < 0 else 'E'

    return f"{degrees:02d}{minutes:.3f}{direction}"


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


def get_float_input(prompt, default_value):
    """
    Prompts the user for a float value with error handling and a default option.
    """
    while True:
        try:
            user_input = input(f"{prompt} (default is {default_value}): ")
            if user_input == "":
                return default_value
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a number.")


def write_cup_file(csv_path, cup_path):
    """
    Reads the filtered CSV file and writes a .cup file with waypoints.
    Args:
        csv_path: Path to the input CSV file.
        cup_path: Path to the output .cup file.
    """
    try:
        with open(csv_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            next(reader)  # Skip the header row

            with open(cup_path, mode='w', newline='') as outfile:
                writer = csv.writer(outfile)
                # Write the standard CUP file header
                writer.writerow(
                    ['Title', 'Code', 'Country', 'Latitude', 'Longitude', 'Elevation', 'Style', 'Direction', 'Length',
                     'Frequency', 'Description'])

                rows_written = 0
                for row in reader:
                    try:
                        # Parse thermal data from the CSV row
                        lat, lon, strength = float(row[0]), float(row[1]), int(float(row[2]))

                        # Create the waypoint name and code from the thermal strength
                        wp_name = str(strength)
                        wp_code = str(strength)

                        # Convert coordinates to the CUP format
                        cup_lat = convert_to_cup_coord(lat, is_lat=True)
                        cup_lon = convert_to_cup_coord(lon, is_lat=False)

                        # Write the new row to the .cup file
                        writer.writerow([
                            wp_name,  # Title
                            wp_code,  # Code
                            'AU',  # Country (defaulting to Australia)
                            cup_lat,  # Latitude
                            cup_lon,  # Longitude
                            '0ft',  # Elevation (placeholder as it's not in the thermal data)
                            '1',  # Style
                            '',  # Direction
                            '',  # Length
                            '',  # Frequency
                            'Filtered Thermal'  # Description
                        ])
                        rows_written += 1
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping malformed row in CSV '{row}'. Error: {e}")

        print(f"\nSuccessfully wrote {rows_written} waypoints to '{cup_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found. Cannot create CUP file.")
    except Exception as e:
        print(f"An unexpected error occurred while writing the CUP file: {e}")


def write_kml_file(csv_path, kml_path):
    """
    Reads the filtered CSV file and writes a KML file with placemarks.
    Args:
        csv_path: Path to the input CSV file.
        kml_path: Path to the output KML file.
    """
    try:
        with open(csv_path, mode='r', newline='') as infile:
            reader = csv.reader(infile)
            next(reader)  # Skip the header row

            with open(kml_path, mode='w', encoding='utf-8') as outfile:
                # Write the KML header
                outfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                outfile.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
                outfile.write('<Document>\n')
                outfile.write('    <name>Filtered Thermals</name>\n')

                rows_written = 0
                for row in reader:
                    try:
                        # Parse thermal data from the CSV row
                        lat, lon, strength = float(row[0]), float(row[1]), int(float(row[2]))

                        # Write the KML Placemark for the thermal
                        outfile.write('    <Placemark>\n')
                        outfile.write(f'        <name>{strength}</name>\n')
                        outfile.write(f'        <description>Thermal strength: {strength}</description>\n')
                        outfile.write('        <Point>\n')
                        # KML coordinates are longitude, latitude, altitude
                        outfile.write(f'            <coordinates>{lon},{lat},0</coordinates>\n')
                        outfile.write('        </Point>\n')
                        outfile.write('    </Placemark>\n')
                        rows_written += 1
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping malformed row in CSV '{row}'. Error: {e}")

                # Write the KML footer
                outfile.write('</Document>\n')
                outfile.write('</kml>\n')

        print(f"\nSuccessfully wrote {rows_written} placemarks to '{kml_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found. Cannot create KML file.")
    except Exception as e:
        print(f"An unexpected error occurred while writing the KML file: {e}")


