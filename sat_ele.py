"""Compatibility API for prior covariance and elevation."""
from gps_processing.processing.geometry import elevation, initial_covariance

def calculate_Cll(result_gps=None, sat3_df=None):
    return initial_covariance()
