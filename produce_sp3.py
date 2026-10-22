"""Compatibility API for SP3 readers. Original sources are in legacy/."""
from gps_processing.io.sp3 import read_sp3, combine_sp3

def process_sp3_file(file_path):
    frame = read_sp3(file_path)
    frame['epok'] = (frame.timestamp-frame.timestamp.min().normalize()).dt.total_seconds()
    return frame

combine_sp3_data = combine_sp3
