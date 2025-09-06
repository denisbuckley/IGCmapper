import os
import datetime
import matplotlib.pyplot as plt
from functions_holder import get_thermals_as_dataframe, consolidate_thermals, get_flight_duration

# ------------------------
# Defaults
# ------------------------
DEFAULT_IGC_FOLDER = "igc"
DEFAULT_TIME_WINDOW = 15
DEFAULT_DISTANCE_THRESHOLD = 100
DEFAULT_ALTITUDE_CHANGE = 20
DEFAULT_MIN_CLIMB_RATE = 0.5
DEFAULT_RADIUS_KM = 2.0

# ------------------------
# Helper for validated input
# ------------------------
def get_input(prompt, default, cast_type):
    while True:
        user_input = input(f"{prompt} (default {default}): ")
        if not user_input:
            return default
        try:
            return cast_type(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a {cast_type.__name__}.")

# ------------------------
# Main workflow
# ------------------------
def main():
    igc_folder = input(f"IGC folder (default '{DEFAULT_IGC_FOLDER}'): ") or DEFAULT_IGC_FOLDER
    igc_folder = os.path.abspath(igc_folder)
    if not os.path.exists(igc_folder):
        print(f"Folder does not exist: {igc_folder}")
        return

    time_window = get_input("Time window (s)", DEFAULT_TIME_WINDOW, int)
    distance_threshold = get_input("Distance threshold (m)", DEFAULT_DISTANCE_THRESHOLD, int)
    altitude_change = get_input("Altitude gain (m)", DEFAULT_ALTITUDE_CHANGE, int)
    min_climb_rate = get_input("Min climb rate (m/s)", DEFAULT_MIN_CLIMB_RATE, float)
    radius_km = get_input("Cluster radius (km)", DEFAULT_RADIUS_KM, float)

    print("\nDetecting thermals...")
    thermal_df = get_thermals_as_dataframe(igc_folder, time_window, distance_threshold, altitude_change)
    print(f"Raw thermals detected: {len(thermal_df)}")

    consolidated_df, coords_df = consolidate_thermals(thermal_df, min_climb_rate, radius_km)
    print(f"Consolidated thermals: {len(consolidated_df)}")

    if not consolidated_df.empty:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        consolidated_df.to_csv(f"consolidated_thermal_data_{timestamp}.csv", index=False)
        coords_df.to_csv(f"consolidated_thermal_coords_{timestamp}.csv", index=False)
        print(f"Saved consolidated thermals CSVs with timestamp {timestamp}")

        # Plot thermals
        plt.figure(figsize=(10,6))
        plt.scatter(coords_df['longitude'], coords_df['latitude'],
                    c=coords_df['strength'], cmap='viridis', s=50, edgecolor='k')
        plt.colorbar(label='Avg Climb Rate (m/s)')
        plt.title('Consolidated Glider Thermals')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.show()

    # Flight durations
    print("\nFlight durations (minutes):")
    for f in os.listdir(igc_folder):
        if f.endswith(".igc"):
            dur = get_flight_duration(os.path.join(igc_folder, f))
            print(f"{f}: {dur:.1f} min")

if __name__ == "__main__":
    main()
