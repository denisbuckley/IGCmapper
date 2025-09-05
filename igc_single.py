"""
Complete, Runnable IGC Thermal & Climb Analysis Script
-------------------------------------------------------
Copy-paste into PyCharm and run. Interactive fallback included.
"""

import os
import sys
import math
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

# ---------------------- Utility Functions ----------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def igc_to_decimal_degrees(igc_coord: str) -> Optional[float]:
    direction = igc_coord[-1].upper()
    body = igc_coord[:-1]
    try:
        if direction in ('N', 'S'):
            deg = int(body[0:2])
            minutes = float(body[2:]) / 1000.0
        else:
            deg = int(body[0:3])
            minutes = float(body[3:]) / 1000.0
    except:
        return None
    dec = deg + minutes / 60.0
    if direction in ('S', 'W'):
        dec = -dec
    return dec


def time_to_seconds(time_str: str) -> Optional[int]:
    try:
        h = int(time_str[0:2])
        m = int(time_str[2:4])
        s = int(time_str[4:6])
        return h * 3600 + m * 60 + s
    except:
        return None

# ---------------------- Parsing ----------------------

def parse_igc_b_record(line: str) -> Optional[tuple]:
    if not line.startswith('B') or len(line) < 35:
        return None
    time_s = time_to_seconds(line[1:7])
    lat = igc_to_decimal_degrees(line[7:15])
    lon = igc_to_decimal_degrees(line[15:24])
    try:
        alt_baro = int(line[25:30])
    except:
        alt_baro = None
    try:
        alt_gps = int(line[30:35])
    except:
        alt_gps = None
    if None in (time_s, lat, lon):
        return None
    return time_s, lat, lon, alt_baro if alt_baro is not None else (alt_gps if alt_gps is not None else 0), alt_gps


def read_igc(filepath: str) -> Dict[str, list]:
    latitudes, longitudes, altitudes, times = [], [], [], []
    pilot_name = 'Unknown Pilot'
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.strip('\n')
            if not line:
                continue
            if line.startswith('H') and line[3:6] == 'PLT':
                pilot_name = line[6:].strip()
                continue
            if not line.startswith('B'):
                continue
            parsed = parse_igc_b_record(line)
            if parsed is None:
                continue
            time_s, lat, lon, alt_baro, alt_gps = parsed
            alt = alt_gps if (alt_gps is not None and 0 < alt_gps < 10000) else alt_baro
            if alt is None:
                continue
            latitudes.append(lat)
            longitudes.append(lon)
            altitudes.append(alt)
            times.append(time_s)
    return {'lat': latitudes, 'lon': longitudes, 'alt': altitudes, 'time': times, 'pilot': pilot_name}

# ---------------------- Analysis ----------------------

def moving_average(a: list, radius: int = 3):
    if radius <= 0:
        return np.array(a)
    kernel = np.ones(2*radius+1)/float(2*radius+1)
    return np.convolve(a, kernel, mode='same')


def detect_thermals(lat, lon, alt, time_s, time_window=10, altitude_change_threshold=20, distance_threshold=100, max_gap_seconds=20):
    n = len(alt)
    circling_idx = []
    for i in range(time_window, n):
        j = i - time_window
        alt_gain = alt[i] - alt[j]
        dist = haversine_distance(lat[i], lon[i], lat[j], lon[j])
        if alt_gain >= altitude_change_threshold and dist <= distance_threshold:
            circling_idx.append(i)
    thermals = []
    if not circling_idx:
        return thermals
    current = {'start_idx': circling_idx[0], 'end_idx': circling_idx[0]}
    for idx in circling_idx[1:]:
        gap = time_s[idx] - time_s[current['end_idx']]
        if idx == current['end_idx'] + 1 or gap <= max_gap_seconds:
            current['end_idx'] = idx
        else:
            thermals.append(current)
            current = {'start_idx': idx, 'end_idx': idx}
    thermals.append(current)
    thermal_out = []
    for t in thermals:
        s, e = t['start_idx'], t['end_idx']
        altitude_gain = alt[e] - alt[s]
        duration = time_s[e] - time_s[s]
        center_lat = np.mean(lat[s:e+1])
        center_lon = np.mean(lon[s:e+1])
        thermal_out.append({'start': s, 'end': e, 'alt_gain': altitude_gain, 'duration_s': duration, 'center_lat': center_lat, 'center_lon': center_lon})
    return thermal_out


def detect_significant_climbs(lat, lon, alt, time_s, significant_climb_threshold=200):
    n = len(alt)
    climbs = []
    in_climb = False
    start = 0
    for i in range(1, n):
        if not in_climb and alt[i] > alt[i-1]:
            in_climb = True
            start = i-1
        elif in_climb and alt[i] <= alt[i-1]:
            end = i-1
            gain = alt[end] - alt[start]
            if gain >= significant_climb_threshold:
                duration = time_s[end] - time_s[start]
                distance = haversine_distance(lat[start], lon[start], lat[end], lon[end])
                climb_rate = gain/duration if duration>0 else 0
                climbs.append({'start': start, 'end': end, 'alt_gain': gain, 'duration_s': duration, 'distance_m': distance, 'climb_rate': climb_rate})
            in_climb = False
    return climbs

# ---------------------- Plotting ----------------------

def plot_flight(lat, lon, alt, thermals, climbs, pilot_name, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(10,8))
    sc = ax.scatter(lon, lat, c=alt, s=8, cmap='viridis', label='Track')
    plt.colorbar(sc, ax=ax, label='Altitude (m)')
    ax.plot(lon, lat, linewidth=0.8)
    ax.scatter([lon[0]], [lat[0]], marker='D', s=80, label='Start')
    ax.scatter([lon[-1]], [lat[-1]], marker='X', s=80, label='End')
    if thermals:
        t_lons = [t['center_lon'] for t in thermals]
        t_lats = [t['center_lat'] for t in thermals]
        ax.scatter(t_lons, t_lats, s=140, facecolors='none', edgecolors='green', linewidths=2, label='Thermals')
    if climbs:
        climb_lons = [lon[c['end']] for c in climbs]
        climb_lats = [lat[c['end']] for c in climbs]
        ax.scatter(climb_lons, climb_lats, marker='^', s=140, facecolors='yellow', edgecolors='k', label='Significant climbs')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'IGC Flight Path for {pilot_name}')
    ax.legend()
    ax.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()

# ---------------------- Wrapper ----------------------

def analyze_igc(filepath):
    data = read_igc(filepath)
    if not data['lat']:
        print('No valid GPS points found.')
        return
    thermals = detect_thermals(data['lat'], data['lon'], data['alt'], data['time'])
    climbs = detect_significant_climbs(data['lat'], data['lon'], data['alt'], data['time'])
    print(f"Pilot: {data['pilot']}")
    print(f"Detected {len(thermals)} thermals and {len(climbs)} significant climbs.")
    plot_flight(data['lat'], data['lon'], data['alt'], thermals, climbs, data['pilot'])

# ---------------------- Main ----------------------

def main():
    filepath = input('Please enter the path to your IGC file: ').strip()
    if not filepath.lower().endswith('.igc'):
        filepath += '.igc'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    analyze_igc(filepath)

if __name__ == '__main__':
    main()
