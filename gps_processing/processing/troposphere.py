"""Standard-atmosphere Saastamoinen zenith delays and Chao mapping functions."""
import numpy as np
from gps_processing.constants import A, B


def geodetic(xyz):
    """ECEF metres -> latitude, longitude (radians), ellipsoidal height (metres)."""
    x, y, z = np.asarray(xyz, dtype=float)
    e2 = 1-(B/A)**2
    p = np.hypot(x, y)
    lat = np.arctan2(z, p*(1-e2))
    for _ in range(6):
        n = A/np.sqrt(1-e2*np.sin(lat)**2)
        height = p/np.cos(lat)-n
        lat = np.arctan2(z, p*(1-e2*n/(n+height)))
    n = A/np.sqrt(1-e2*np.sin(lat)**2)
    return lat, np.arctan2(y, x), p/np.cos(lat)-n


class Troposphere:
    def __init__(self, xyz, humidity=0.5):
        lat, _, height = geodetic(xyz)
        height = float(np.clip(height, -500, 9000))
        pressure = 1013.25*(1-2.2557e-5*height)**5.2568
        temperature = 15-6.5e-3*height+273.15
        vapour = 6.108*humidity*np.exp((17.15*temperature-4684)/(temperature-38.45))
        self.zhd = 0.0022768*pressure/(1-0.00266*np.cos(2*lat)-0.00028*height/1000)
        self.zwd = 0.002277*(1255/temperature+0.05)*vapour

    @staticmethod
    def mapping(elevation_deg):
        e = np.radians(np.maximum(elevation_deg, 1.0))
        dry = 1/(np.sin(e)+0.00143/(np.tan(e)+0.0445))
        wet = 1/(np.sin(e)+0.00035/(np.tan(e)+0.017))
        return dry, wet
