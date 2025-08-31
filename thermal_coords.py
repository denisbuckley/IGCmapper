#
# This script reads IGC files from a specified folder, uses functions from
# the functions_holder module to process the data, and returns consolidated
# DataFrames for easy export and analysis. This version is interactive and
# prompts the user for analysis parameters, providing default values.
# detailed description at end of script


# DataFrame with strength saved to 'consolidated_thermal_data.csv'
# DataFrame with only coordinates saved to 'consolidated_thermal_coords.csv'

# Required libraries: pandas, numpy
# Make sure your functions_holder.py script is in the same directory.
#
# Defaults
# Time Window = 15s: The script analyzes data in a rolling time window.
# For a set of consecutive data points, if the duration of those points is within this time window (15 seconds in your example), it proceeds with the next checks.
# It's essentially looking for a sustained period of time where the other conditions are met.
# Distance Threshold = 100m: Within the defined time window, the script checks if the total distance traveled is less than or equal to this threshold.
# A short distance traveled over a period of time indicates that the glider is circling in one spot, which is a classic sign of a thermal.

# Altitude Gain = 20m: Within the time window, the script checks if the glider's altitude has increased by at least this amount.
# This is a crucial parameter, as a thermal is an area of rising air, so you would expect to gain altitude when flying through one.
#
# Once all the individual "thermals" are detected using the parameters above, these next two parameters are used to group and filter them.
# The goal is to combine multiple detected thermals that are actually the same thermal and to filter out any that aren't very useful for the glider pilot.

# Min Climb Rate = 0.5 m/s: This parameter acts as a quality filter. The script calculates the average climb rate for each detected thermal.
# If the climb rate is below this threshold, the thermal is discarded from the final output.
# Gliders are constantly sinking, so a thermal that offers less than 0.5 m/s of lift might not be worth circling in.
#
# Radius = 2.0 km: This parameter is used to consolidate or group the thermals.
# The script looks for thermals that are located within this radius of each other.
# If it finds a cluster of thermals within a 2.0 km radius, it treats them all as a single, stronger thermal at a central location.
# This helps to clean up the data and prevents you from seeing multiple thermal markers for what was likely the same thermal that the glider circled a few times.

import os
import pandas as pd
import numpy as np
from functions_holder import get_thermals_as_dataframe, consolidate_thermals

if __name__ == "__main__":
    # --- User-configurable parameters with default values ---
    # Define the default values here.
    default_igc_folder = "./igc"
    default_time_window = 10
    default_distance_threshold = 100
    default_altitude_change_threshold = 20
    default_min_climb_rate = 0.5
    default_radius_km = 1.0

    # Get input from the user, using defaults if they just press Enter.
    try:
        # Folder path
        igc_folder_input = input(
            f"Please enter the path to the folder with IGC files (default: {default_igc_folder}): ")
        igc_folder = igc_folder_input if igc_folder_input else default_igc_folder

        # Heuristic parameters for identifying a thermal
        time_window_input = input(
            f"Enter time window for thermal detection in seconds (default: {default_time_window}): ")
        time_window = int(time_window_input) if time_window_input else default_time_window

        distance_threshold_input = input(
            f"Enter max distance traveled in meters (default: {default_distance_threshold}): ")
        distance_threshold = int(distance_threshold_input) if distance_threshold_input else default_distance_threshold

        altitude_change_threshold_input = input(
            f"Enter min altitude gain in meters (default: {default_altitude_change_threshold}): ")
        altitude_change_threshold = int(
            altitude_change_threshold_input) if altitude_change_threshold_input else default_altitude_change_threshold

        # Parameters for consolidating thermals
        min_climb_rate_input = input(
            f"Enter min climb rate in m/s to filter thermals (default: {default_min_climb_rate}): ")
        min_climb_rate = float(min_climb_rate_input) if min_climb_rate_input else default_min_climb_rate

        radius_km_input = input(f"Enter radius in km to consolidate thermals (default: {default_radius_km}): ")
        radius_km = float(radius_km_input) if radius_km_input else default_radius_km

    except ValueError:
        print("\nInvalid input. Please enter a number for the numeric parameters. Exiting.")
        exit()

    print("\nRunning thermal data analysis with the following parameters:")
    print(f"IGC Folder: {igc_folder}")
    print(
        f"Thermal Detection: Time Window = {time_window}s, Distance Threshold = {distance_threshold}m, Altitude Gain = {altitude_change_threshold}m")
    print(f"Consolidation: Min Climb Rate = {min_climb_rate} m/s, Radius = {radius_km} km")

    # Step 1: Get the raw thermal data as a DataFrame
    # Now passing the user-defined parameters to the function
    thermal_df = get_thermals_as_dataframe(igc_folder, time_window, distance_threshold, altitude_change_threshold)
    print(f"Initial raw thermal count: {len(thermal_df)}")

    # Step 2: Consolidate the thermals based on the user-defined logic
    consolidated_df, coords_df = consolidate_thermals(thermal_df, min_climb_rate, radius_km)
    print(f"Consolidated thermal count after filtering and grouping: {len(consolidated_df)}")

    # Check if the final DataFrame is empty before attempting to print/save
    if not consolidated_df.empty:
        print("\n--- Consolidated Thermal Data DataFrame (with strength) ---")
        print(consolidated_df)

        print("\n--- Consolidated Thermal Coordinates DataFrame (without strength) ---")
        print(coords_df)

        # You can save the DataFrames to CSV files for use in Google Earth or other tools
        output_csv_with_strength = "consolidated_thermal_data.csv"
        consolidated_df.to_csv(output_csv_with_strength, index=False)
        print(f"\nDataFrame with strength saved to '{output_csv_with_strength}'")

        output_csv_coords = "consolidated_thermal_coords.csv"
        coords_df.to_csv(output_csv_coords, index=False)
        print(f"DataFrame with only coordinates saved to '{output_csv_coords}'")
    else:
        print("\nNo thermals were found that met all the specified criteria. Please try adjusting your parameters.")

'''this script is designed to be an interactive tool for analyzing flight data from gliders. 
Its main purpose is to automatically identify and consolidate thermals—areas of rising air—from IGC flight log files.

Here is a breakdown of what the script does:

Interactive Setup and Data Input
First, the script prompts the user for a series of parameters, 
such as the folder where the IGC files are located and the specific values to use for thermal detection. 
It provides default values for each parameter, so you can simply press Enter to accept the recommended settings. 
This makes the script flexible and easy to use.

Thermal Detection and Consolidation
The script performs two main analysis steps, which are likely handled by a separate file named functions_holder.py:

Initial Thermal Detection: It reads each IGC file and scans the flight data. 
Using the parameters you provide, such as time window, distance threshold, and altitude gain, 
it identifies every possible segment of the flight where the glider was likely circling in a thermal.

Consolidation and Filtering: After finding all the initial "raw" thermal segments, 
it consolidates them into a cleaner, more useful dataset. It does this by:

Filtering out weak thermals: It discards any thermals that have a climb rate below the specified minimum (e.g., 0.5 m/s).

Grouping close thermals: It looks for clusters of detected thermals that are close to each other 
(e.g., within a 2.0 km radius) and combines them into a single, consolidated thermal entry. 
This prevents the same thermal from showing up multiple times in the final results.

Output and Export
Finally, the script provides a summary of its findings, showing the number of thermals detected before and after consolidation. 
It then saves the final, processed data into two CSV files (consolidated_thermal_data.csv and consolidated_thermal_coords.csv). 
These files are in a format that can be easily used with other tools like mapping software or spreadsheet programs for further analysis.



'''