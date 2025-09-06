"""
Enhanced IGC Thermal & Climb Analysis Script
--------------------------------------------
Improved version with error corrections and enhanced features.
"""

import os
import sys
import math
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, NamedTuple
import numpy as np

# Handle optional dependencies gracefully
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------- Configuration ----------------------

@dataclass
class AnalysisConfig:
    """Configuration parameters for IGC analysis"""
    thermal_time_window: int = 10
    thermal_altitude_threshold: float = 20.0
    thermal_distance_threshold: float = 100.0
    significant_climb_threshold: float = 200.0
    max_gap_seconds: int = 20
    vario_threshold: float = 1.0  # m/s minimum climb rate for thermals
    max_file_size_mb: int = 50
    altitude_bounds: Tuple[int, int] = (-500, 15000)
    moving_avg_radius: int = 3

# ---------------------- Data Structures ----------------------

@dataclass
class ThermalInfo:
    """Information about a detected thermal"""
    start_idx: int
    end_idx: int
    altitude_gain: float
    duration_seconds: int
    center_lat: float
    center_lon: float
    average_climb_rate: float
    max_climb_rate: float

@dataclass
class ClimbInfo:
    """Information about a significant climb"""
    start_idx: int
    end_idx: int
    altitude_gain: float
    duration_seconds: int
    distance_meters: float
    average_climb_rate: float
    max_climb_rate: float

@dataclass
class FlightStats:
    """Comprehensive flight statistics"""
    total_distance: float
    max_altitude: float
    min_altitude: float
    total_climb: float
    total_sink: float
    average_speed: float
    max_speed: float
    flight_duration: int
    pilot_name: str

# ---------------------- Custom Exceptions ----------------------

class IGCParseError(Exception):
    """Raised when IGC file cannot be parsed"""
    pass

class IGCValidationError(Exception):
    """Raised when IGC data validation fails"""
    pass

# ---------------------- Utility Functions ----------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using haversine formula"""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def haversine_distance_vectorized(lat1: np.ndarray, lon1: np.ndarray,
                                lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance calculation"""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (np.sin(dphi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def igc_to_decimal_degrees(igc_coord: str) -> Optional[float]:
    """Convert IGC coordinate format to decimal degrees"""
    if len(igc_coord) < 4:
        return None

    direction = igc_coord[-1].upper()
    body = igc_coord[:-1]

    try:
        if direction in ('N', 'S'):
            if len(body) < 7:
                return None
            deg = int(body[0:2])
            minutes = float(body[2:]) / 1000.0  # Already in decimal minutes
        else:  # E, W
            if len(body) < 8:
                return None
            deg = int(body[0:3])
            minutes = float(body[3:]) / 1000.0  # Already in decimal minutes
    except (ValueError, IndexError):
        return None

    dec = deg + minutes / 60.0
    if direction in ('S', 'W'):
        dec = -dec
    return dec

def time_to_seconds(time_str: str) -> Optional[int]:
    """Convert HHMMSS time string to seconds since midnight"""
    if len(time_str) != 6:
        return None
    try:
        h = int(time_str[0:2])
        m = int(time_str[2:4])
        s = int(time_str[4:6])
        if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
            return None
        return h * 3600 + m * 60 + s
    except ValueError:
        return None

def validate_track_point(lat: float, lon: float, alt: float, time_s: int,
                        prev_time: Optional[int] = None,
                        config: AnalysisConfig = AnalysisConfig()) -> bool:
    """Validate GPS point for reasonableness"""
    # Check coordinate bounds
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return False

    # Check altitude bounds
    if not (config.altitude_bounds[0] <= alt <= config.altitude_bounds[1]):
        return False

    # Check time progression
    if prev_time is not None and time_s <= prev_time:
        return False

    return True

def moving_average(data: np.ndarray, radius: int = 3) -> np.ndarray:
    """Apply moving average smoothing"""
    if radius <= 0:
        return data
    kernel = np.ones(2 * radius + 1) / float(2 * radius + 1)
    return np.convolve(data, kernel, mode='same')

# ---------------------- Parsing Functions ----------------------

def parse_igc_b_record(line: str) -> Optional[Tuple[int, float, float, float, Optional[float]]]:
    """Parse IGC B-record (GPS fix)"""
    if not line.startswith('B') or len(line) < 35:
        return None

    try:
        time_s = time_to_seconds(line[1:7])
        lat = igc_to_decimal_degrees(line[7:15])
        lon = igc_to_decimal_degrees(line[15:24])

        # Parse altitudes
        alt_baro = None
        alt_gps = None

        try:
            alt_baro = int(line[25:30])
        except (ValueError, IndexError):
            pass

        try:
            alt_gps = int(line[30:35])
        except (ValueError, IndexError):
            pass

        if None in (time_s, lat, lon):
            return None

        # Prefer GPS altitude if reasonable, fall back to barometric
        alt = alt_gps if (alt_gps is not None and 0 < alt_gps < 10000) else alt_baro
        if alt is None:
            alt = 0

        return time_s, lat, lon, alt, alt_gps

    except Exception as e:
        raise IGCParseError(f"Invalid B-record format: {line[:20]}... Error: {e}")

def read_igc(filepath: str, config: AnalysisConfig = AnalysisConfig()) -> Dict[str, np.ndarray]:
    """Read and parse IGC file with validation"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"IGC file not found: {filepath}")

    # Check file size
    file_size = os.path.getsize(filepath)
    if file_size > config.max_file_size_mb * 1024 * 1024:
        raise IGCValidationError(f"IGC file too large: {file_size / (1024*1024):.1f}MB")

    logger.info(f"Reading IGC file: {filepath} ({file_size / 1024:.1f}KB)")

    latitudes, longitudes, altitudes, times = [], [], [], []
    pilot_name = 'Unknown Pilot'
    glider_type = 'Unknown'
    date = 'Unknown'
    prev_time = None

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            for line_num, raw_line in enumerate(fh, 1):
                line = raw_line.strip()
                if not line:
                    continue

                # Parse header information
                if line.startswith('H'):
                    if 'PLT' in line:
                        pilot_name = line.split('PLT')[1].strip(':').strip()
                    elif 'GTY' in line:
                        glider_type = line.split('GTY')[1].strip(':').strip()
                    elif 'DTE' in line:
                        date = line.split('DTE')[1].strip(':').strip()
                    continue

                if not line.startswith('B'):
                    continue

                try:
                    parsed = parse_igc_b_record(line)
                    if parsed is None:
                        continue

                    time_s, lat, lon, alt, alt_gps = parsed

                    # Validate point
                    if not validate_track_point(lat, lon, alt, time_s, prev_time, config):
                        logger.warning(f"Invalid track point at line {line_num}: {lat:.6f}, {lon:.6f}, {alt}m")
                        continue

                    latitudes.append(lat)
                    longitudes.append(lon)
                    altitudes.append(alt)
                    times.append(time_s)
                    prev_time = time_s

                except IGCParseError as e:
                    logger.warning(f"Parse error at line {line_num}: {e}")
                    continue

    except Exception as e:
        raise IGCParseError(f"Error reading IGC file: {e}")

    if not latitudes:
        raise IGCValidationError("No valid GPS points found in IGC file")

    logger.info(f"Parsed {len(times)} GPS points for pilot: {pilot_name}")

    return {
        'lat': np.array(latitudes),
        'lon': np.array(longitudes),
        'alt': np.array(altitudes),
        'time': np.array(times),
        'pilot': pilot_name,
        'glider': glider_type,
        'date': date
    }

# ---------------------- Analysis Functions ----------------------

def calculate_climb_rates(alt: np.ndarray, time: np.ndarray,
                         smooth_radius: int = 3) -> np.ndarray:
    """Calculate climb rates (variometer readings)"""
    # Smooth altitude data
    alt_smooth = moving_average(alt, smooth_radius)

    # Calculate time differences
    dt = np.diff(time)
    dt = np.where(dt == 0, 1, dt)  # Avoid division by zero

    # Calculate climb rates
    climb_rates = np.diff(alt_smooth) / dt

    # Pad to match original length
    climb_rates = np.concatenate([[climb_rates[0]], climb_rates])

    return climb_rates

def detect_thermals_enhanced(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
                           time: np.ndarray, config: AnalysisConfig = AnalysisConfig()) -> List[ThermalInfo]:
    """Enhanced thermal detection using climb rate and circling behavior"""
    n = len(alt)
    if n < config.thermal_time_window * 2:
        return []

    # Calculate climb rates
    climb_rates = calculate_climb_rates(alt, time, config.moving_avg_radius)

    # Find potential thermal points
    thermal_candidates = []

    for i in range(config.thermal_time_window, n):
        j = i - config.thermal_time_window

        # Check altitude gain
        alt_gain = alt[i] - alt[j]
        if alt_gain < config.thermal_altitude_threshold:
            continue

        # Check horizontal distance (indicates circling)
        dist = haversine_distance(lat[i], lon[i], lat[j], lon[j])
        if dist > config.thermal_distance_threshold:
            continue

        # Check average climb rate in window
        avg_climb_rate = np.mean(climb_rates[j:i+1])
        if avg_climb_rate < config.vario_threshold:
            continue

        thermal_candidates.append(i)

    if not thermal_candidates:
        return []

    # Group consecutive points into thermals
    thermals = []
    current_start = thermal_candidates[0]
    current_end = thermal_candidates[0]

    for idx in thermal_candidates[1:]:
        gap = time[idx] - time[current_end]
        if idx == current_end + 1 or gap <= config.max_gap_seconds:
            current_end = idx
        else:
            # Finalize current thermal
            thermal = _create_thermal_info(current_start, current_end, lat, lon, alt, time, climb_rates)
            if thermal:
                thermals.append(thermal)
            current_start = idx
            current_end = idx

    # Don't forget the last thermal
    thermal = _create_thermal_info(current_start, current_end, lat, lon, alt, time, climb_rates)
    if thermal:
        thermals.append(thermal)

    logger.info(f"Detected {len(thermals)} thermals")
    return thermals

def _create_thermal_info(start: int, end: int, lat: np.ndarray, lon: np.ndarray,
                        alt: np.ndarray, time: np.ndarray, climb_rates: np.ndarray) -> Optional[ThermalInfo]:
    """Create ThermalInfo object from indices"""
    if end <= start:
        return None

    altitude_gain = alt[end] - alt[start]
    duration = time[end] - time[start]

    if duration <= 0:
        return None

    center_lat = np.mean(lat[start:end+1])
    center_lon = np.mean(lon[start:end+1])
    avg_climb_rate = altitude_gain / duration
    max_climb_rate = np.max(climb_rates[start:end+1])

    return ThermalInfo(
        start_idx=start,
        end_idx=end,
        altitude_gain=altitude_gain,
        duration_seconds=duration,
        center_lat=center_lat,
        center_lon=center_lon,
        average_climb_rate=avg_climb_rate,
        max_climb_rate=max_climb_rate
    )

def detect_significant_climbs(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
                            time: np.ndarray, config: AnalysisConfig = AnalysisConfig()) -> List[ClimbInfo]:
    """Detect significant climbs with improved logic"""
    n = len(alt)
    if n < 2:
        return []

    climbs = []
    climb_rates = calculate_climb_rates(alt, time, config.moving_avg_radius)

    in_climb = False
    start_idx = 0

    for i in range(1, n):
        if not in_climb and alt[i] > alt[i-1]:
            # Start of potential climb
            in_climb = True
            start_idx = i - 1
        elif in_climb and alt[i] <= alt[i-1]:
            # End of climb
            end_idx = i - 1
            climb = _create_climb_info(start_idx, end_idx, lat, lon, alt, time, climb_rates, config)
            if climb:
                climbs.append(climb)
            in_climb = False

    # Handle climb continuing to end of flight
    if in_climb:
        end_idx = n - 1
        climb = _create_climb_info(start_idx, end_idx, lat, lon, alt, time, climb_rates, config)
        if climb:
            climbs.append(climb)

    logger.info(f"Detected {len(climbs)} significant climbs")
    return climbs

def _create_climb_info(start: int, end: int, lat: np.ndarray, lon: np.ndarray,
                      alt: np.ndarray, time: np.ndarray, climb_rates: np.ndarray,
                      config: AnalysisConfig) -> Optional[ClimbInfo]:
    """Create ClimbInfo object from indices"""
    if end <= start:
        return None

    altitude_gain = alt[end] - alt[start]
    if altitude_gain < config.significant_climb_threshold:
        return None

    duration = time[end] - time[start]
    if duration <= 0:
        return None

    distance = haversine_distance(lat[start], lon[start], lat[end], lon[end])
    avg_climb_rate = altitude_gain / duration
    max_climb_rate = np.max(climb_rates[start:end+1])

    return ClimbInfo(
        start_idx=start,
        end_idx=end,
        altitude_gain=altitude_gain,
        duration_seconds=duration,
        distance_meters=distance,
        average_climb_rate=avg_climb_rate,
        max_climb_rate=max_climb_rate
    )

def calculate_flight_statistics(data: Dict[str, np.ndarray]) -> FlightStats:
    """Calculate comprehensive flight statistics"""
    lat, lon, alt, time = data['lat'], data['lon'], data['alt'], data['time']
    n = len(lat)

    # Calculate total distance
    if n > 1:
        distances = haversine_distance_vectorized(lat[:-1], lon[:-1], lat[1:], lon[1:])
        total_distance = np.sum(distances)

        # Calculate speeds
        time_diffs = np.diff(time)
        time_diffs = np.where(time_diffs == 0, 1, time_diffs)  # Avoid division by zero
        speeds = distances / time_diffs
        avg_speed = total_distance / (time[-1] - time[0]) if time[-1] > time[0] else 0
        max_speed = np.max(speeds) if len(speeds) > 0 else 0
    else:
        total_distance = 0
        avg_speed = 0
        max_speed = 0

    # Calculate altitude changes
    alt_changes = np.diff(alt)
    total_climb = np.sum(alt_changes[alt_changes > 0])
    total_sink = abs(np.sum(alt_changes[alt_changes < 0]))

    return FlightStats(
        total_distance=total_distance,
        max_altitude=np.max(alt),
        min_altitude=np.min(alt),
        total_climb=total_climb,
        total_sink=total_sink,
        average_speed=avg_speed,
        max_speed=max_speed,
        flight_duration=time[-1] - time[0] if len(time) > 1 else 0,
        pilot_name=data.get('pilot', 'Unknown')
    )

# ---------------------- Plotting Functions ----------------------

def plot_flight(data: Dict[str, np.ndarray], thermals: List[ThermalInfo],
               climbs: List[ClimbInfo], stats: FlightStats,
               save_path: Optional[str] = None, show: bool = True):
    """Plot flight path with thermals and climbs"""
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available. Cannot create plot.")
        return

    lat, lon, alt = data['lat'], data['lon'], data['alt']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Main flight path plot
    sc = ax1.scatter(lon, lat, c=alt, s=8, cmap='viridis', alpha=0.7, label='Track')
    ax1.plot(lon, lat, linewidth=0.5, alpha=0.5, color='gray')

    # Start and end points
    ax1.scatter([lon[0]], [lat[0]], marker='D', s=100, color='green',
               edgecolor='black', label='Start', zorder=5)
    ax1.scatter([lon[-1]], [lat[-1]], marker='X', s=100, color='red',
               edgecolor='black', label='End', zorder=5)

    # Thermal locations as black X markers
    if thermals:
        t_lons = [t.center_lon for t in thermals]
        t_lats = [t.center_lat for t in thermals]

        ax1.scatter(t_lons, t_lats, marker='x', s=200, color='black',
                   linewidths=3, label='Thermals', zorder=4)

    # Significant climbs as yellow lines
    if climbs:
        for i, climb in enumerate(climbs):
            start_idx = climb.start_idx
            end_idx = climb.end_idx
            climb_lons = lon[start_idx:end_idx+1]
            climb_lats = lat[start_idx:end_idx+1]

            # Plot yellow line for the climb segment
            label = 'Significant Climbs' if i == 0 else ""
            ax1.plot(climb_lons, climb_lats, color='yellow', linewidth=4,
                    alpha=0.8, solid_capstyle='round', label=label, zorder=3)

    # Colorbar and labels
    plt.colorbar(sc, ax=ax1, label='Altitude (m)')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'Flight Path - {stats.pilot_name} ({data.get("date", "Unknown Date")})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Altitude profile
    time_hours = (data['time'] - data['time'][0]) / 3600
    ax2.plot(time_hours, alt, linewidth=1, color='blue', label='Altitude')

    # Mark thermals on altitude profile
    if thermals:
        for thermal in thermals:
            start_time = (data['time'][thermal.start_idx] - data['time'][0]) / 3600
            end_time = (data['time'][thermal.end_idx] - data['time'][0]) / 3600
            ax2.axvspan(start_time, end_time, alpha=0.3, color='orange', label='Thermal' if thermal == thermals[0] else "")

    ax2.set_xlabel('Flight Time (hours)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude Profile')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")

    if show:
        plt.show()

def print_analysis_summary(stats: FlightStats, thermals: List[ThermalInfo],
                          climbs: List[ClimbInfo]):
    """Print comprehensive analysis summary"""
    print("\n" + "="*60)
    print("FLIGHT ANALYSIS SUMMARY")
    print("="*60)

    print(f"Pilot: {stats.pilot_name}")
    print(f"Flight Duration: {stats.flight_duration // 3600:02d}:{(stats.flight_duration % 3600) // 60:02d}:{stats.flight_duration % 60:02d}")
    print(f"Total Distance: {stats.total_distance / 1000:.2f} km")
    print(f"Average Speed: {stats.average_speed * 3.6:.1f} km/h")
    print(f"Maximum Speed: {stats.max_speed * 3.6:.1f} km/h")
    print(f"Altitude Range: {stats.min_altitude:.0f}m - {stats.max_altitude:.0f}m")
    print(f"Total Climb: {stats.total_climb:.0f}m")
    print(f"Total Sink: {stats.total_sink:.0f}m")

    print(f"\nThermal Analysis ({len(thermals)} thermals detected):")
    if thermals:
        total_thermal_gain = sum(t.altitude_gain for t in thermals)
        avg_thermal_gain = total_thermal_gain / len(thermals)
        best_thermal = max(thermals, key=lambda t: t.altitude_gain)

        print(f"  Total altitude gained in thermals: {total_thermal_gain:.0f}m")
        print(f"  Average gain per thermal: {avg_thermal_gain:.0f}m")
        print(f"  Best thermal: {best_thermal.altitude_gain:.0f}m in {best_thermal.duration_seconds//60}:{best_thermal.duration_seconds%60:02d}")
        print(f"  Best climb rate: {best_thermal.max_climb_rate:.1f} m/s")

    print(f"\nSignificant Climbs ({len(climbs)} detected):")
    if climbs:
        total_climb_gain = sum(c.altitude_gain for c in climbs)
        best_climb = max(climbs, key=lambda c: c.altitude_gain)

        print(f"  Total gain in significant climbs: {total_climb_gain:.0f}m")
        print(f"  Best climb: {best_climb.altitude_gain:.0f}m over {best_climb.distance_meters/1000:.1f}km")
        print(f"  Best climb rate: {best_climb.max_climb_rate:.1f} m/s")

# ---------------------- Main Analysis Function ----------------------

def analyze_igc(filepath: str, config: AnalysisConfig = AnalysisConfig(),
               save_plot: Optional[str] = None, show_plot: bool = True):
    """Complete IGC analysis workflow"""
    try:
        # Read and parse IGC file
        data = read_igc(filepath, config)

        # Perform analysis
        thermals = detect_thermals_enhanced(data['lat'], data['lon'], data['alt'], data['time'], config)
        climbs = detect_significant_climbs(data['lat'], data['lon'], data['alt'], data['time'], config)
        stats = calculate_flight_statistics(data)

        # Print summary
        print_analysis_summary(stats, thermals, climbs)

        # Create plot
        if HAS_MATPLOTLIB and (show_plot or save_plot):
            plot_flight(data, thermals, climbs, stats, save_plot, show_plot)

        return {
            'data': data,
            'thermals': thermals,
            'climbs': climbs,
            'stats': stats
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

# ---------------------- Command Line Interface ----------------------

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Analyze IGC flight files for thermals and climbs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python igc_analyzer.py flight.igc
  python igc_analyzer.py flight.igc --output analysis.png --no-display
  python igc_analyzer.py flight.igc --thermal-threshold 30 --climb-threshold 150
        """
    )

    parser.add_argument('igc_file', help='Path to IGC file')
    parser.add_argument('--output', '-o', help='Save plot to file (PNG format)')
    parser.add_argument('--no-display', action='store_true', help="Don't show plot on screen")
    parser.add_argument('--thermal-threshold', type=float, default=20.0,
                       help='Minimum altitude gain for thermal detection (default: 20m)')
    parser.add_argument('--climb-threshold', type=float, default=200.0,
                       help='Minimum altitude gain for significant climb (default: 200m)')
    parser.add_argument('--thermal-distance', type=float, default=100.0,
                       help='Maximum horizontal distance for thermal (default: 100m)')
    parser.add_argument('--vario-threshold', type=float, default=1.0,
                       help='Minimum climb rate for thermal (default: 1.0 m/s)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    return parser

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate file path
    filepath = args.igc_file
    if not filepath.lower().endswith('.igc'):
        filepath += '.igc'

    if not os.path.exists(filepath):
        print(f"Error: IGC file not found: {filepath}")
        sys.exit(1)

    # Create configuration
    config = AnalysisConfig(
        thermal_altitude_threshold=args.thermal_threshold,
        significant_climb_threshold=args.climb_threshold,
        thermal_distance_threshold=args.thermal_distance,
        vario_threshold=args.vario_threshold
    )

    try:
        # Perform analysis
        analyze_igc(filepath, config, args.output, not args.no_display)

    except (IGCParseError, IGCValidationError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            raise
        sys.exit(1)

def interactive_main():
    """Interactive mode for backward compatibility"""
    print("IGC Flight Analysis Tool")
    print("-" * 30)

    while True:
        filepath = input('\nPlease enter the path to your IGC file (or "quit" to exit): ').strip()

        if filepath.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        if not filepath.lower().endswith('.igc'):
            filepath += '.igc'

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        try:
            print(f"\nAnalyzing {filepath}...")
            analyze_igc(filepath)

            # Ask if user wants to analyze another file
            again = input("\nAnalyze another flight? (y/n): ").strip().lower()
            if again not in ('y', 'yes'):
                break

        except Exception as e:
            print(f"Analysis failed: {e}")
            logger.exception("Detailed error information:")

if __name__ == '__main__':
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Fall back to interactive mode
        interactive_main()