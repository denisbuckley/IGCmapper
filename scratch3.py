# Purpose:
# This is the main, executable script of the application. Its role is to
# handle user interaction, orchestrate the data flow between other modules,
# and present the final results to the user. It is the only file the
# user should run directly.

import os
import sys
import pandas as pd
import numpy as np
from thermal_analyzer import get_thermals_as_dataframe, consolidate_thermals

if __name__ == "__main__":
    print("Welcome to the Thermal IGC Analysis tool.")
    print("-" * 35)

    # --- Logic for requesting the folder name, with a default value ---
    igc_folder = input("Enter the path to the folder containing your IGC files (default: 'igc'): ") or "igc"

    if not os.path.isdir(igc_folder):
        print(f"\nError: The specified folder '{igc_folder}' does not exist.")
        sys.exit()

    print("\nRunning thermal data analysis with the following parameters:")

    # --- User-configurable parameters with original default values ---
    default_time_window = 10
    default_distance_threshold = 100
    default_altitude_change_threshold = 20
    default_min_climb_rate = 0.5
    default_radius_km = 1.0

    try:
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
        sys.exit()

    print("\nRunning thermal data analysis with the following parameters:")
    print(f"IGC Folder: {igc_folder}")
    print(
        f"Thermal Detection: Time Window = {time_window}s, Distance Threshold = {distance_threshold}m, Altitude Gain = {altitude_change_threshold}m")
    print(f"Consolidation: Min Climb Rate = {min_climb_rate} m/s, Radius = {radius_km} km")

    # Step 1: Get the raw thermal data as a DataFrame
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
