"""Standard-atmosphere Saastamoinen zenith delays; Niell (1996) or Chao (1972) mapping functions.

Niell: hydrostatic and wet continued fractions (Marini form) with latitude-interpolated
coefficients, an annual term for the hydrostatic part (phase DOY 28, +half a year in the
southern hemisphere) and the hydrostatic height correction. It needs only latitude, height and
day of year, as in Shi & Gao (2012); without a date the annual term is left out.
"""
import numpy as np
from gps_processing.constants import A, B

MAPPINGS = ('niell', 'chao')
NIELL_LATITUDES = np.array([15., 30., 45., 60., 75.])
NIELL_HYDROSTATIC_MEAN = np.array([[1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3],
                                   [2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3],
                                   [62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3]])
NIELL_HYDROSTATIC_AMPLITUDE = np.array([[0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5],
                                        [0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5],
                                        [0.0, 9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5]])
NIELL_HEIGHT = (2.53e-5, 5.49e-3, 1.14e-3)
NIELL_WET = np.array([[5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4],
                      [1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3],
                      [4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2]])


def marini(sine, a, b, c):
    """Continued fraction normalised to 1 at the zenith."""
    return (1+a/(1+b/(1+c)))/(sine+a/(sine+b/(sine+c)))


def chao_mapping(elevation_deg):
    e = np.radians(np.maximum(elevation_deg, 1.0))
    dry = 1/(np.sin(e)+0.00143/(np.tan(e)+0.0445))
    wet = 1/(np.sin(e)+0.00035/(np.tan(e)+0.017))
    return dry, wet


def niell_mapping(elevation_deg, latitude, height, doy=None):
    """Niell (1996) hydrostatic and wet mapping; ``latitude`` in radians, ``height`` in metres."""
    sine = np.sin(np.radians(np.maximum(elevation_deg, 1.0)))
    lat = np.degrees(abs(latitude))
    interpolate = lambda table: np.array([np.interp(lat, NIELL_LATITUDES, row) for row in table])
    hydrostatic = interpolate(NIELL_HYDROSTATIC_MEAN)
    if doy is not None:
        phase = doy-28+(182.625 if latitude < 0 else 0.0)
        hydrostatic = hydrostatic-interpolate(NIELL_HYDROSTATIC_AMPLITUDE)*np.cos(2*np.pi*phase/365.25)
    dry = marini(sine, *hydrostatic)+(1/sine-marini(sine, *NIELL_HEIGHT))*height/1000
    return dry, marini(sine, *interpolate(NIELL_WET))


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
    def __init__(self, xyz, humidity=0.5, mapping='niell', doy=None):
        if mapping not in MAPPINGS:
            raise ValueError(f'Mapping function must be one of {", ".join(MAPPINGS)}.')
        lat, _, height = geodetic(xyz)
        self.mapping_name, self.doy, self.latitude, self.height = mapping, doy, lat, height
        height = float(np.clip(height, -500, 9000))
        pressure = 1013.25*(1-2.2557e-5*height)**5.2568
        temperature = 15-6.5e-3*height+273.15
        vapour = 6.108*humidity*np.exp((17.15*temperature-4684)/(temperature-38.45))
        self.zhd = 0.0022768*pressure/(1-0.00266*np.cos(2*lat)-0.00028*height/1000)
        self.zwd = 0.002277*(1255/temperature+0.05)*vapour

    def mapping(self, elevation_deg):
        if self.mapping_name == 'chao':
            return chao_mapping(elevation_deg)
        return niell_mapping(elevation_deg, self.latitude, self.height, self.doy)
