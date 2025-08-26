import csv


def generate_cup_file(input_csv_path, output_cup_path):
    """
    Reads a CSV file with thermal coordinates and strength,
    and writes a .cup file for flight navigation.

    Args:
        input_csv_path (str): The path to the input CSV file.
        output_cup_path (str): The path for the output .cup file.
    """
    try:
        # Open the input CSV file for reading
        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)

            # Check if the required columns are in the CSV
            if 'latitude' not in reader.fieldnames or \
                    'longitude' not in reader.fieldnames or \
                    'strength' not in reader.fieldnames:
                print("Error: The CSV file must contain 'latitude', 'longitude', and 'strength' columns.")
                return

            # Open the output .cup file for writing
            with open(output_cup_path, mode='w', newline='', encoding='utf-8') as outfile:
                # Write the header line for the .cup file
                # The format is 'name,code,country,lat,lon,elev,style,desc'
                # We'll use strength as the name and description.
                outfile.write("name,code,country,lat,lon,elev,style,desc\n")

                # Iterate over each row in the CSV file
                for row in reader:
                    # Extract the data from the row
                    # The 'strength' column is used for the waypoint name and description
                    # The 'latitude' and 'longitude' columns are used for coordinates
                    name = row['strength']
                    latitude = row['latitude']
                    longitude = row['longitude']

                    # Assuming a standard decimal degree format for coordinates
                    # The elev (elevation) and style are set to 0 and 'TRP' (Turn Point)
                    elev = "0.0"
                    style = "TRP"
                    desc = f"Thermal Strength: {row['strength']}"

                    # Format the line for the .cup file.
                    # Note: .cup files can handle various coordinate formats.
                    # This script uses standard decimal degrees.
                    cup_line = f"{name},,{latitude},{longitude},{elev},{style},{desc}\n"

                    # Write the formatted line to the output file
                    outfile.write(cup_line)

        print(f"Successfully created '{output_cup_path}' from '{input_csv_path}'.")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Main script execution ---
if __name__ == "__main__":
    # Define the input and output file paths
    # Replace 'consolidated_thermal_coords.csv' with your actual file name
    input_file = "consolidated_thermal_coords.csv"
    output_file = "thermal_waypoints.cup"

    # Run the function to generate the .cup file
    generate_cup_file(input_file, output_file)
