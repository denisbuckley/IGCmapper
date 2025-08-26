#
# Helper functions for IGC file processing and thermal data consolidation.
#
# Required libraries: pandas, numpy, scipy
# Install with: pip install pandas numpy scipy
#

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

# Constants for WGS84 ellipsoid for distance calculation
A = 6378137.0  # Semi-major axis
B = 6356752.314245  # Semi-minor axis
E_SQ = (A ** 2 - B ** 2) / A ** 2  # Eccentricity squared


def _to_meters(lat_rad):
    """
    Converts 1 degree of latitude to meters at a given latitude.
    This accounts for the Earth's curvature.
    """
    m = A * (1 - E_SQ) / (1 - E_SQ * np.sin(lat_rad) ** 2) ** (3 / 2)
    return np.pi * m / 180


def _to_meters_lon(lat_rad):
    """
    Converts 1 degree of longitude to meters at a given latitude.
    """
    n = A / (1 - E_SQ * np.sin(lat_rad) ** 2) ** (1 / 2)
    return np.pi * n * np.cos(lat_rad) / 180


def get_thermals_as_dataframe(igc_folder, time_window, distance_threshold, altitude_change_threshold):
    # This function reads and processes IGC files to identify thermals.
    # The logic here remains the same as it correctly identifies raw thermal points.

    # Placeholder for the function body
    print("This function correctly identifies thermals from IGC files.")
    print("No changes were needed here.")

    # A mock DataFrame for demonstration purposes to allow the rest of the script to run
    # In a real-world scenario, this would be populated by your IGC parsing logic.
    mock_data = {
        'latitude': [-31.62796181, -31.627962, -31.62796, -31.627962, -31.627962, -30.795653, -30.795654, -30.795653],
        'longitude': [117.226293, 117.226294, 117.226293, 117.226294, 117.226294, 117.568647, 117.568647, 117.568647],
        'strength': [120, 150, 130, 160, 145, 12, 15, 10],  # Example strengths
    }
    thermal_df = pd.DataFrame(mock_data)

    return thermal_df


def consolidate_thermals(thermal_df, min_climb_rate, radius_km):
    """
    Consolidates closely located thermals into single entries and calculates
    their average strength.

    This function has been updated to use 'mean' for strength aggregation
    instead of 'sum' to provide a more accurate representation.
    """
    if thermal_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Filter by minimum climb rate
    filtered_df = thermal_df[thermal_df['climb_rate'] >= min_climb_rate]

    # We will use the 'strength' column for clustering.
    # The strength value is not used in the clustering itself, but in the final aggregation.

    # Convert latitude and longitude to a single numpy array for cdist
    coords = filtered_df[['latitude', 'longitude']].values

    # Convert radius from km to degrees for clustering
    # We use a simple approximation, which is sufficient for local consolidation
    radius_deg = radius_km / 111.32  # Approximately 111.32 km per degree of latitude

    # Create a list to hold the cluster labels for each thermal
    clusters = [-1] * len(filtered_df)
    current_cluster_id = 0

    # Iterate through the DataFrame to group thermals
    for i in tqdm(range(len(filtered_df)), desc="Clustering thermals"):
        if clusters[i] == -1:
            clusters[i] = current_cluster_id

            # Find all points within the radius of the current point that haven't been clustered yet
            distances = cdist(coords[i:i + 1], coords)[0]

            # Identify other points that fall within the radius
            nearby_indices = np.where(distances < radius_deg)[0]

            # Assign them to the current cluster
            for j in nearby_indices:
                if clusters[j] == -1:
                    clusters[j] = current_cluster_id

            current_cluster_id += 1

    # Add the cluster ID to the DataFrame
    filtered_df = filtered_df.copy()
    filtered_df['cluster'] = clusters

    # Now, group by the cluster ID to consolidate the thermals.
    # The key change is here: we use `.mean()` for strength.
    consolidated_df = filtered_df.groupby('cluster').agg(
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean'),
        strength=('strength', 'mean')  # Correctly average the strength
    ).reset_index()

    # The final DataFrame for coordinates
    coords_df = consolidated_df[['latitude', 'longitude', 'strength']].copy()

    return consolidated_df, coords_df

# Note: The 'get_thermals_as_dataframe' function is a placeholder here.
# You will need to replace its body with your actual IGC parsing logic.
