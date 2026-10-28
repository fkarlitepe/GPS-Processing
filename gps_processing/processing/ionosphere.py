"""Global ionosphere map (IONEX) evaluation and the second-order ionospheric term.

* Single-layer model: ionospheric pierce point on a thin shell at the IONEX height,
  slant factor F = 1/sqrt(1-(R cos e/(R+H))^2)  (IONEX "COSZ" mapping).
* VTEC: bilinear interpolation in latitude/longitude and linear interpolation between
  the two nearest maps after rotating them with the Sun (IONEX 1.0, eq. 3).
* Second order: s = 7527 c B cos(theta) STEC (Bassiri & Hajj 1993; Hernandez-Pajares
  et al. 2007; IERS Conventions 2010, sec. 9.4) with a thin-shell approximation and a
  centred-dipole geomagnetic field (IGRF-13, epoch 2020). The first-order term is
  removed by the ionosphere-free combination, so this residual is what a GIM adds to
  dual-frequency PPP. With code delay s/f^3 and phase advance -s/(2 f^3) per frequency,
  the IF code delay is -s/(f1 f2 (f1+f2)) and the IF phase delay +s/(2 f1 f2 (f1+f2)).
"""
import numpy as np
from gps_processing.constants import C, F1, F2

TECU = 1e16
FIRST_ORDER = 40.3                          # m^3/s^2: first-order delay = 40.3 STEC/f^2
DIPOLE_NT = np.array([-1450.9, 4652.5, -29404.8])   # (g11, h11, g10), IGRF-13 2020, nT
DIPOLE_RADIUS = 6371.2e3


def pierce_point(lat, lon, elevation, azimuth, radius_km, height_km):
    """Return IPP latitude/longitude (radians) and the single-layer slant factor."""
    rp = radius_km/(radius_km+height_km)*np.cos(elevation)
    ap = np.pi/2-elevation-np.arcsin(rp)
    lat_p = np.arcsin(np.sin(lat)*np.cos(ap)+np.cos(lat)*np.sin(ap)*np.cos(azimuth))
    over_pole = ((lat > np.radians(70)) & (np.tan(ap)*np.cos(azimuth) > np.tan(np.pi/2-lat))) | \
                ((lat < -np.radians(70)) & (-np.tan(ap)*np.cos(azimuth) > np.tan(np.pi/2+lat)))
    swing = np.arcsin(np.clip(np.sin(ap)*np.sin(azimuth)/np.cos(lat_p), -1, 1))
    lon_p = np.where(over_pole, lon+np.pi-swing, lon+swing)
    return lat_p, (lon_p+np.pi) % (2*np.pi)-np.pi, 1/np.sqrt(1-rp**2)


def bilinear(maps, grid, lat_deg, lon_deg):
    lats, lons = maps.latitudes, maps.longitudes
    dlat, dlon = lats[1]-lats[0], lons[1]-lons[0]
    lon_deg = (lon_deg-lons[0]) % 360+lons[0]
    fi = np.clip((lat_deg-lats[0])/dlat, 0, len(lats)-1-1e-9)
    fj = np.clip((lon_deg-lons[0])/dlon, 0, len(lons)-1-1e-9)
    i, j = np.floor(fi).astype(int), np.floor(fj).astype(int)
    p, q = fi-i, fj-j
    return ((1-p)*(1-q)*grid[i, j]+(1-p)*q*grid[i, j+1]+p*(1-q)*grid[i+1, j]+p*q*grid[i+1, j+1])


def vtec(maps, epoch, lat_deg, lon_deg):
    """VTEC (TECU) at seconds-of-day ``epoch`` with Sun-fixed rotation between maps."""
    t = maps.epochs
    tolerance = (t[1]-t[0]) if len(t) > 1 else 3600
    if epoch < t[0]-tolerance or epoch > t[-1]+tolerance:
        raise ValueError('IONEX maps do not cover the observation epochs.')
    k = int(np.clip(np.searchsorted(t, epoch)-1, 0, max(len(t)-2, 0)))
    if len(t) == 1:
        return bilinear(maps, maps.tec[0], lat_deg, lon_deg)
    t0, t1 = t[k], t[k+1]
    w = float(np.clip((epoch-t0)/(t1-t0), 0, 1))
    e0 = bilinear(maps, maps.tec[k], lat_deg, lon_deg+(epoch-t0)*360/86400)
    e1 = bilinear(maps, maps.tec[k+1], lat_deg, lon_deg+(epoch-t1)*360/86400)
    return (1-w)*e0+w*e1


def dipole_field(position):
    """Centred-dipole geomagnetic field (Tesla) at ECEF positions (n, 3) in metres."""
    r = np.linalg.norm(position, axis=1, keepdims=True)
    unit = position/r
    return (DIPOLE_RADIUS/r)**3*(3*(unit@DIPOLE_NT)[:, None]*unit-DIPOLE_NT)*1e-9


def ionosphere_free_second_order(stec_tecu, field, direction):
    """IF code and phase second-order delays (m). ``direction``: unit vectors satellite -> receiver."""
    s = 7527*C*np.einsum('ij,ij->i', field, direction)*stec_tecu*TECU
    scale = F1*F2*(F1+F2)
    return -s/scale, s/(2*scale)
