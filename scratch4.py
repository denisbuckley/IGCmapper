"""
Fixed Thermal Detection Debug Tool
----------------------------------
Now using the exact Gemini algorithm that actually works.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def parse_b_record_enhanced(line: str):
    """Enhanced B-record parsing"""
    if not line.startswith('B') or len(line) < 35:
        return None

    try:
        time_str = line[1:7]
        lat_str = line[7:15]
        lon_str = line[15:24]

        try:
            gps_alt = int(line[30:35])
        except (ValueError, IndexError):
            try:
                gps_alt = int(line[25:30])
            except (ValueError, IndexError):
                return None

        # Fixed coordinate parsing
        lat = int(lat_str[:2]) + int(lat_str[2:7]) / 60000.0
        if lat_str[7] == 'S':
            lat = -lat

        lon = int(lon_str[:3]) + int(lon_str[3:8]) / 60000.0
        if lon_str[8] == 'W':
            lon = -lon

        try:
            time = datetime.datetime.strptime(time_str, '%H%M%S').time()
        except ValueError:
            return None

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None

        if not (-500 <= gps_alt <= 15000):
            return None

        return {
            'time': time,
            'latitude': lat,
            'longitude': lon,
            'altitude': gps_alt
        }
    except Exception:
        return None

def read_igc_file_enhanced(file_path: str):
    """Read IGC file and extract track points and metadata"""
    points = []
    metadata = {'pilot': 'Unknown', 'glider': 'Unknown', 'date': 'Unknown'}

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            for line in file:
                line = line.strip()

                if line.startswith('H'):
                    if 'PLT' in line or 'PILOT' in line:
                        metadata['pilot'] = line.split(':')[-1].strip() if ':' in line else 'Unknown'
                    elif 'GTY' in line or 'GLIDER' in line:
                        metadata['glider'] = line.split(':')[-1].strip() if ':' in line else 'Unknown'
                    elif 'DTE' in line:
                        metadata['date'] = line.split(':')[-1].strip() if ':' in line else 'Unknown'

                elif line.startswith('B'):
                    record = parse_b_record_enhanced(line)
                    if record:
                        points.append(record)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], metadata

    return points, metadata

def get_flight_duration_enhanced(points):
    """Calculate flight duration in minutes"""
    if len(points) < 2:
        return 0.0

    first_time = points[0]['time']
    last_time = points[-1]['time']

    first_seconds = first_time.hour * 3600 + first_time.minute * 60 + first_time.second
    last_seconds = last_time.hour * 3600 + last_time.minute * 60 + last_time.second

    if last_seconds < first_seconds:
        last_seconds += 24 * 3600

    return (last_seconds - first_seconds) / 60.0

def detect_thermals_single_flight(points, time_window=15, distance_threshold=100,
                                 altitude_change_threshold=20, min_climb_rate=0.5):
    """
    Detect thermals using the original Gemini algorithm that actually works.
    """
    if len(points) < 2:
        return []

    df = pd.DataFrame(points)
    df['timestamp'] = df['time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)

    # Handle overnight flights
    for i in range(1, len(df)):
        if df.loc[i, 'timestamp'] < df.loc[i-1, 'timestamp']:
            df.loc[i:, 'timestamp'] += 24 * 3600

    # Vectorized haversine distance calculation (same as Gemini)
    lat_rad = np.radians(df['latitude'].values)
    lon_rad = np.radians(df['longitude'].values)
    R = 6371000  # meters

    delta_lat = lat_rad[1:] - lat_rad[:-1]
    delta_lon = lon_rad[1:] - lon_rad[:-1]
    a = np.sin(delta_lat/2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(delta_lon/2)**2
    distances = 2 * R * np.arcsin(np.sqrt(a))
    distances = np.insert(distances, 0, 0)  # distance from first point = 0

    thermals = []

    # Rolling time window detection - EXACT GEMINI ALGORITHM
    start_idx = 0
    for i in range(len(df)):
        while df.loc[i, 'timestamp'] - df.loc[start_idx, 'timestamp'] > time_window:
            start_idx += 1

        dist_window = np.sum(distances[start_idx:i+1])
        alt_gain = df.loc[i, 'altitude'] - df.loc[start_idx, 'altitude']
        duration = df.loc[i, 'timestamp'] - df.loc[start_idx, 'timestamp']

        # GEMINI'S LOGIC: Only check distance and altitude - NO climb rate filter here!
        if dist_window <= distance_threshold and alt_gain >= altitude_change_threshold and duration > 0:
            climb_rate = alt_gain / duration
            thermals.append({
                'start_time': df.loc[start_idx, 'time'],
                'start_lat': df.loc[start_idx, 'latitude'],
                'start_lon': df.loc[start_idx, 'longitude'],
                'center_lat': df.loc[start_idx:i+1, 'latitude'].mean(),
                'center_lon': df.loc[start_idx:i+1, 'longitude'].mean(),
                'altitude_change': alt_gain,
                'duration_s': duration,
                'avg_climb_rate_mps': climb_rate,
                'window_distance': dist_window
            })

    # Apply climb rate filter AFTER detection (like Gemini's consolidate_thermals)
    if min_climb_rate > 0:
        thermals = [t for t in thermals if t['avg_climb_rate_mps'] >= min_climb_rate]

    return thermals

def debug_single_flight(igc_file_path: str, show_plots: bool = False):
    """Debug thermal detection for a single flight"""
    print(f"\n{'='*60}")
    print(f"DEBUG ANALYSIS: {os.path.basename(igc_file_path)}")
    print(f"{'='*60}")

    # Read flight data
    points, metadata = read_igc_file_enhanced(igc_file_path)

    if not points:
        print("No valid GPS points found in file")
        return

    print(f"Successfully read {len(points)} GPS points")
    print(f"   Pilot: {metadata['pilot']}")
    print(f"   Glider: {metadata['glider']}")
    print(f"   Date: {metadata['date']}")

    # Flight duration
    duration = get_flight_duration_enhanced(points)
    print(f"   Flight duration: {duration:.1f} minutes")

    if duration < 30:
        print("Flight might be too short for thermal detection")

    # Convert to dataframe for analysis
    df = pd.DataFrame(points)
    df['timestamp'] = df['time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second)

    # Handle overnight flights
    for i in range(1, len(df)):
        if df.loc[i, 'timestamp'] < df.loc[i-1, 'timestamp']:
            df.loc[i:, 'timestamp'] += 24 * 3600

    # Calculate basic statistics
    print(f"\nFLIGHT STATISTICS:")
    print(f"   Altitude range: {df['altitude'].min():.0f}m - {df['altitude'].max():.0f}m")
    print(f"   Max altitude gain: {df['altitude'].max() - df['altitude'].min():.0f}m")

    # Calculate climb rates
    alt_diff = df['altitude'].diff()
    time_diff = df['timestamp'].diff()
    climb_rates = alt_diff / time_diff
    climb_rates = climb_rates.fillna(0)

    positive_climbs = climb_rates[climb_rates > 0]
    if len(positive_climbs) > 0:
        print(f"   Average climb rate (climbing sections): {positive_climbs.mean():.2f} m/s")
        print(f"   Max climb rate: {positive_climbs.max():.2f} m/s")
        print(f"   Percentage time climbing: {len(positive_climbs)/len(climb_rates)*100:.1f}%")
    else:
        print("   No positive climb rates found!")

    # Test different detection parameters using GEMINI'S ALGORITHM
    print(f"\nTESTING THERMAL DETECTION PARAMETERS (Using Gemini Algorithm):")

    parameter_sets = [
        {"name": "Default (Gemini)", "time_window": 15, "distance_threshold": 100, "altitude_change_threshold": 20, "min_climb_rate": 0.5},
        {"name": "Relaxed", "time_window": 30, "distance_threshold": 200, "altitude_change_threshold": 15, "min_climb_rate": 0.3},
        {"name": "Moderate", "time_window": 20, "distance_threshold": 150, "altitude_change_threshold": 10, "min_climb_rate": 0.2},
        {"name": "Very Strict", "time_window": 45, "distance_threshold": 300, "altitude_change_threshold": 30, "min_climb_rate": 1.0},
        {"name": "No Climb Filter", "time_window": 15, "distance_threshold": 100, "altitude_change_threshold": 20, "min_climb_rate": 0.0},
    ]

    best_count = 0
    best_result = None
    results = []

    for params in parameter_sets:
        print(f"\n   Testing {params['name']} parameters...")
        print(f"      Time window: {params['time_window']}s")
        print(f"      Distance threshold: {params['distance_threshold']}m")
        print(f"      Altitude threshold: {params['altitude_change_threshold']}m")
        print(f"      Min climb rate: {params['min_climb_rate']} m/s")

        # Create parameters dict without the 'name' key for the function call
        func_params = {k: v for k, v in params.items() if k != 'name'}

        try:
            thermals = detect_thermals_single_flight(points, **func_params)
            thermal_count = len(thermals)
            print(f"      → {thermal_count} thermals found")

            result = {
                'name': params['name'],
                'count': thermal_count,
                'params': func_params,
                'thermals': thermals
            }
            results.append(result)

            if thermal_count > best_count:
                best_count = thermal_count
                best_result = result

        except Exception as e:
            print(f"      → Error: {e}")
            result = {
                'name': params['name'],
                'count': 0,
                'params': func_params,
                'thermals': []
            }
            results.append(result)

    # Show results summary
    print(f"\nPARAMETER TEST RESULTS:")
    for result in results:
        status = "✓" if result['count'] > 0 else "✗"
        print(f"   {status} {result['name']}: {result['count']} thermals")

    if best_count == 0:
        print(f"\nNO THERMALS DETECTED WITH ANY PARAMETER SET")

        # Analyze flight characteristics
        if duration > 300:  # > 5 hours
            print("This appears to be a CROSS-COUNTRY FLIGHT")
        if positive_climbs.max() > 10:
            print("Very high climb rates suggest WAVE or RIDGE SOARING")
        if df['altitude'].max() - df['altitude'].min() > 2000:
            print("Large altitude gains suggest XC or mountain flying")

        print("\nRecommendations:")
        print("- Try a shorter LOCAL SOARING flight (1-3 hours)")
        print("- Look for flights with traditional thermal circling")
        print("- Check GPS logging interval (should be ≤5 seconds)")

        return

    print(f"\nBEST RESULTS: {best_count} thermals found")
    print(f"Best parameter set: {best_result['name']}")
    print(f"\nRecommended settings for thermal mapping:")
    for key, value in best_result['params'].items():
        print(f"  --{key.replace('_', '-')}: {value}")

    # Get detailed thermal info with best parameters
    thermals = best_result['thermals']

    print(f"\nTHERMAL DETAILS:")
    for i, thermal in enumerate(thermals, 1):
        print(f"   Thermal {i}:")
        print(f"      Location: {thermal['center_lat']:.6f}, {thermal['center_lon']:.6f}")
        print(f"      Altitude gain: {thermal['altitude_change']:.0f}m")
        print(f"      Duration: {thermal['duration_s']:.0f}s")
        print(f"      Climb rate: {thermal['avg_climb_rate_mps']:.2f} m/s")

    print(f"\nAnalysis complete! Found {len(thermals)} thermals with {best_result['name']} parameters.")

def debug_folder(igc_folder: str):
    """Debug thermal detection for all flights in folder"""
    if not os.path.exists(igc_folder):
        print(f"Folder not found: {igc_folder}")
        return

    igc_files = [f for f in os.listdir(igc_folder) if f.lower().endswith('.igc')]
    if not igc_files:
        print(f"No IGC files found in {igc_folder}")
        return

    print(f"DEBUGGING {len(igc_files)} IGC FILES")
    print(f"Folder: {igc_folder}")

    total_thermals = 0
    valid_flights = 0

    for filename in igc_files[:10]:  # Limit to first 10 files
        file_path = os.path.join(igc_folder, filename)

        # Quick analysis without plots
        points, metadata = read_igc_file_enhanced(file_path)

        if not points:
            print(f"✗ {filename}: No valid points")
            continue

        duration = get_flight_duration_enhanced(points)
        if duration < 30:
            print(f"✗ {filename}: Too short ({duration:.1f} min)")
            continue

        # Test with Gemini default parameters
        thermals = detect_thermals_single_flight(
            points,
            time_window=15,
            distance_threshold=100,
            altitude_change_threshold=20,
            min_climb_rate=0.5
        )

        total_thermals += len(thermals)
        valid_flights += 1

        print(f"✓ {filename}: {len(thermals)} thermals, {duration:.1f} min, Pilot: {metadata['pilot']}")

    print(f"\nSUMMARY:")
    print(f"Valid flights: {valid_flights}/{min(len(igc_files), 10)}")
    print(f"Total thermals found: {total_thermals}")
    print(f"Average thermals per flight: {total_thermals/valid_flights if valid_flights > 0 else 0:.1f}")
    print(f"Parameters used: Gemini Default (15s window, 100m distance, 20m altitude, 0.5 m/s)")

def main():
    """Simple prompt-based debug tool for PyCharm"""
    print("THERMAL DETECTION DEBUG TOOL (Using Gemini Algorithm)")
    print("=" * 60)

    # Simple input prompt - no loops
    choice = input("Enter 1 for single file, 2 for folder: ").strip()

    if choice == '1':
        file_path = input("Enter IGC file path: ").strip()
        if os.path.exists(file_path):
            debug_single_flight(file_path, show_plots=False)
        else:
            print(f"ERROR: File not found: {file_path}")

    elif choice == '2':
        folder_path = input("Enter IGC folder path: ").strip()
        debug_folder(folder_path)

    else:
        print("Invalid choice. Please run again and enter 1 or 2.")

    print("\nAnalysis complete. Script finished.")

if __name__ == "__main__":
    main()