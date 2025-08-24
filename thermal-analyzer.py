#
# This script analyzes multiple IGC files to determine the probability distributions
# of thermal distance and strength. It now imports the core functions from the
# 'functions_holder' module, making the code more modular and reusable.
#
# Required libraries: matplotlib, numpy
# Make sure your functions_holder.py script is in the same directory.
#
# Install with: pip install matplotlib numpy
#

import matplotlib.pyplot as plt
import numpy as np
import os
# Import all the core functions from the new holder file
import functions_holder


def main():
    """
    Main function to analyze multiple IGC files and plot thermal distributions.
    """
    # --- Input folder to analyze ---
    folder_path = input("Please enter the path to the folder containing your IGC files: ")

    # Filter for IGC files
    igc_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.igc')]

    if not igc_files:
        print(f"No IGC files found in the folder: {folder_path}. Please check the path and try again.")
        return

    all_thermal_strengths = []
    all_thermal_distances = []
    total_flight_duration_seconds = 0
    total_flight_distance_meters = 0

    print("Starting multi-file thermal analysis...")

    for filename in igc_files:
        print(f"Processing file: {filename}")

        # Call the function from the new module
        thermals, duration_s, distance_m = functions_holder.find_thermals_in_file(filename)

        total_flight_duration_seconds += duration_s
        total_flight_distance_meters += distance_m

        if not thermals:
            continue

        # Collect thermal strengths
        for thermal in thermals:
            # Only consider thermals with a positive climb rate
            if thermal['climb_rate'] > 0:
                all_thermal_strengths.append(thermal['climb_rate'])

        # Calculate and collect distances between successive thermals for this flight
        for i in range(1, len(thermals)):
            start_lat1, start_lon1 = thermals[i - 1]['start_location']
            start_lat2, start_lon2 = thermals[i]['start_location']
            # Call the haversine_distance function from the new module
            distance = functions_holder.haversine_distance(start_lat1, start_lon1, start_lat2, start_lon2)
            all_thermal_distances.append(distance / 1000)  # Store in kilometers

    print("\n--- Summary of all analyzed IGC files ---")
    total_thermals = len(all_thermal_strengths)
    print(f"Total thermals identified across all files: {total_thermals}")

    if not all_thermal_strengths:
        print("No thermals were identified in any of the provided files. Cannot perform analysis.")
        return

    # --- Print Summary Statistics ---
    print("\n--- Thermal Strength Distribution ---")
    average_strength = np.mean(all_thermal_strengths)
    print(f"Average Strength: {average_strength:.2f} m/s")
    print(f"Median Strength: {np.median(all_thermal_strengths):.2f} m/s")
    print(f"Standard Deviation: {np.std(all_thermal_strengths):.2f} m/s")

    # Calculate lambda for the exponential distribution
    exponential_lambda = 1 / average_strength if average_strength > 0 else 0
    print(f"Calculated Exponential lambda (λ) parameter: {exponential_lambda:.4f}")

    print("\n--- Distance Between Thermals Distribution ---")
    if all_thermal_distances:
        average_distance_km = np.mean(all_thermal_distances)
        print(f"Average Distance: {average_distance_km:.2f} km")
        print(f"Median Distance: {np.median(all_thermal_distances):.2f} km")
        print(f"Standard Deviation: {np.std(all_thermal_distances):.2f} km")

        # Calculate the probability of encountering a thermal per kilometer
        thermal_rate_per_km = 1 / average_distance_km if average_distance_km > 0 else 0
        print(f"Calculated Thermal Encounter Rate per km: {thermal_rate_per_km:.2f} thermals/km")
    else:
        print("Not enough thermals to calculate distances.")

    print("\n--- Poisson Distribution Analysis ---")
    total_flight_distance_km = total_flight_distance_meters / 1000
    if total_flight_distance_km > 0:
        poisson_lambda = total_thermals / total_flight_distance_km
        print(f"Total flight distance: {total_flight_distance_km:.2f} km")
        print(f"Calculated Poisson lambda (λ) parameter: {poisson_lambda:.2f} thermals/km")
    else:
        print("Not enough flight data to calculate Poisson lambda.")

    # --- Plot the distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Thermal Strength Distribution
    axes[0].hist(all_thermal_strengths, bins=15, density=True, color='skyblue', edgecolor='black')
    axes[0].set_title('Probability Distribution of Thermal Strength')
    axes[0].set_xlabel('Thermal Strength (Average Climb Rate in m/s)')
    axes[0].set_ylabel('Probability Density')
    axes[0].grid(axis='y', alpha=0.75)

    # Plot 2: Distance Between Thermals Distribution
    if all_thermal_distances:
        axes[1].hist(all_thermal_distances, bins=15, density=True, color='lightgreen', edgecolor='black')
        axes[1].set_title('Probability Distribution of Thermal Distance')
        axes[1].set_xlabel('Distance Between Thermals (km)')
        axes[1].set_ylabel('Probability Density')
        axes[1].grid(axis='y', alpha=0.75)
    else:
        axes[1].set_title('Not enough thermals to plot distances')

    plt.suptitle('Thermal Distribution Analysis Across All Flights')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
