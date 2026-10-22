"""Compatibility API for the sequential float solution."""
from gps_processing.processing.ppp import solve

def process_result_gps(result_gps, sat3_df, cll_result, approximate_xyz=None):
    if approximate_xyz is None:
        approximate_xyz = result_gps.attrs.get('approximate_xyz')
    if approximate_xyz is None:
        raise ValueError('Yaklaşık XYZ koordinatını explicit olarak geçin.')
    positions, ambiguities, _, _, correction = solve(result_gps,sat3_df,approximate_xyz,covariance=cll_result)
    return positions[['x','y','z']].tail(1), ambiguities, correction.reshape(-1,1)
