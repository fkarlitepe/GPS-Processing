"""Compatibility API; explicit metadata replaces hidden Excel files."""
from gps_processing.io.rinex import read_rinex

def process_gps_file(rinex_file_path):
    obs = read_rinex(rinex_file_path)
    frame = obs.combined.copy()
    frame.attrs['approximate_xyz'] = obs.approximate_xyz
    frame.attrs['raw'] = obs.raw
    return frame
