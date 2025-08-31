# This script acts as a command-line interface to the `find_thermals_in_file` and
# `get_flight_duration` functions.
# It prompts the user for a filename and the necessary thermal-detection parameters,
# then calls the functions and prints the results.
#
# Both files, 'igc_parser.py' and 'functions_holder.py', should be in the same directory.
from functions_holder import find_thermals_in_file, get_flight_duration


def main():
    """
    Main function to get user input, find thermals, get flight duration, and print the results.
    """
    print("Welcome to the IGC Thermal Finder!")

    # Use a loop to allow the user to try again if a file is not found
    while True:
        try:
            filename = input("\nPlease enter the name of the IGC file (e.g., 'flight_log.igc'): ")

            # Get the three required parameters from the user with new default values
            # The time window to look for stable climbs
            time_window_str = input(
                "Enter the time window in seconds (e.g., 30) or press Enter for default (20): "
            )
            time_window = float(time_window_str) if time_window_str else 20.0

            # The minimum horizontal distance covered in a thermal
            distance_threshold_str = input(
                "Enter the distance threshold in meters (e.g., 100) or press Enter for default (100.0): "
            )
            distance_threshold = float(distance_threshold_str) if distance_threshold_str else 100.0

            # The minimum altitude change to qualify as a thermal
            altitude_change_threshold_str = input(
                "Enter the altitude change threshold in meters (e.g., 50) or press Enter for default (20.0): "
            )
            altitude_change_threshold = float(altitude_change_threshold_str) if altitude_change_threshold_str else 20.0

            # Call the function to find thermals
            thermals_found = find_thermals_in_file(
                filename,
                time_window,
                distance_threshold,
                altitude_change_threshold
            )

            # Call the function to get the flight duration
            duration = get_flight_duration(filename)

            # Count the number of thermals found
            num_thermals = len(thermals_found)

            print(f"\nSuccessfully processed '{filename}'.")
            print(f"Found {num_thermals} thermals in the file.")
            print(f"The total flight duration was {duration:.2f} minutes.")

            break  # Exit the loop if successful

        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found. Please check the name and try again.")
        except ValueError:
            print("Error: Invalid input. Please enter a valid number for the parameters.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


# This ensures the code inside the block only runs when the script is executed directly
if __name__ == "__main__":
    main()
