"""Solid Earth tide station displacement, IERS Conventions (2010) eq. 7.5, degree 2, step 1.

    dr = sum_{Moon, Sun} (GM_j/GM_E) (R_E^4/R_j^3) [ h2 r (3/2 (Rj.r)^2 - 1/2) + 3 l2 (Rj.r)(Rj - (Rj.r) r) ]

with unit vectors r (station) and Rj (body), h2 = 0.6078, l2 = 0.0847 (no latitude, degree-3 or
frequency-dependent terms: they are at the millimetre level). The full displacement including
the permanent part is applied, i.e. coordinates are in the conventional tide-free system used
by the IGS. Moon: low-precision series of Montenbruck & Gill (2000, sec. 3.3.2); Sun: see
processing.antenna.sun_position. The radial tide reaches ~30 cm and changes by decimetres within
hours, so without it float PPP ambiguities absorb a time-varying range error.

Also here (IERS Conventions 2010, chapter 7; all applied as range corrections on IF code and phase):
* ocean tide loading: 11 constituents from a BLQ file with the astronomical arguments of the
  IERS ARG routine (as in RTKLIB tide_oload); up to a few cm near coasts, semi-diurnal;
* pole tide: eq. 7.24 with the IERS 2018 linear mean pole; a few mm, nearly constant over a day;
* Shapiro delay: relativistic signal delay 2GM/c^2 ln((rs+rr+rho)/(rs+rr-rho)), ~2 cm and
  elevation dependent; IGS clocks assume the user applies it.
"""
from datetime import datetime
import numpy as np
from gps_processing.constants import C
from gps_processing.processing.antenna import sun_position, enu_axes

GM = 3.986004418e14
# Ocean-loading constituents (BLQ order): angular speed (rad/s) and multipliers of the
# astronomical arguments H0 (Sun), S0 (Moon), P0 (lunar perigee) plus a phase in cycles.
OCEAN_ARGUMENTS = np.array([
    [1.40519e-4, 2.0, -2.0, 0.0, 0.00],    # M2
    [1.45444e-4, 0.0, 0.0, 0.0, 0.00],     # S2
    [1.37880e-4, 2.0, -3.0, 1.0, 0.00],    # N2
    [1.45842e-4, 2.0, 0.0, 0.0, 0.00],     # K2
    [0.72921e-4, 1.0, 0.0, 0.0, 0.25],     # K1
    [0.67598e-4, 1.0, -2.0, 0.0, -0.25],   # O1
    [0.72523e-4, -1.0, 0.0, 0.0, -0.25],   # P1
    [0.64959e-4, 1.0, -3.0, 1.0, -0.25],   # Q1
    [0.53234e-5, 0.0, 2.0, 0.0, 0.00],     # Mf
    [0.26392e-5, 0.0, 1.0, -1.0, 0.00],    # Mm
    [0.03982e-5, 2.0, 0.0, 0.0, 0.00]])    # Ssa

R_EARTH = 6378136.6
GM_RATIO = {'moon': 0.012300037, 'sun': 332946.0482}
H2, L2 = 0.6078, 0.0847


def moon_position(epochs, date):
    """Moon ECEF position (m), accuracy of a few arcminutes and ~500 km in distance."""
    n = (date-datetime(2000, 1, 1, 12)).total_seconds()/86400+np.asarray(epochs, float)/86400
    t = n/36525
    rad = np.radians
    mean = 218.31617+481267.88088*t-1.3972*t
    l, lp = rad(134.96292+477198.86753*t), rad(357.52543+35999.04944*t)
    f, d = rad(93.27283+483202.01873*t), rad(297.85027+445267.11135*t)
    longitude = mean+(22640*np.sin(l)+769*np.sin(2*l)-4586*np.sin(l-2*d)+2370*np.sin(2*d)-668*np.sin(lp)
                      - 412*np.sin(2*f)-212*np.sin(2*l-2*d)-206*np.sin(l+lp-2*d)+192*np.sin(l+2*d)
                      - 165*np.sin(lp-2*d)+148*np.sin(l-lp)-125*np.sin(d)-110*np.sin(l+lp)-55*np.sin(2*f-2*d))/3600
    latitude = (18520*np.sin(f+rad(longitude-mean)+rad((412*np.sin(2*f)+541*np.sin(lp))/3600))
                - 526*np.sin(f-2*d)+44*np.sin(l+f-2*d)-31*np.sin(-l+f-2*d)-25*np.sin(-2*l+f)
                - 23*np.sin(lp+f-2*d)+21*np.sin(-l+f)+11*np.sin(-lp+f-2*d))/3600
    distance = 1e3*(385000-20905*np.cos(l)-3699*np.cos(2*d-l)-2956*np.cos(2*d)-570*np.cos(2*l)
                    + 246*np.cos(2*l-2*d)-205*np.cos(lp-2*d)-171*np.cos(l+2*d)-152*np.cos(l+lp-2*d))
    lam, beta, eps = rad(longitude), rad(latitude), rad(23.43929111-0.0130042*t)
    x = distance*np.cos(beta)*np.cos(lam)
    y = distance*np.cos(beta)*np.sin(lam)
    z = distance*np.sin(beta)
    y, z = np.cos(eps)*y-np.sin(eps)*z, np.sin(eps)*y+np.cos(eps)*z     # ecliptic -> equator
    theta = rad(280.46061837+360.98564736629*n)
    return np.column_stack([np.cos(theta)*x+np.sin(theta)*y, -np.sin(theta)*x+np.cos(theta)*y, z])


def solid_tide(receiver_xyz, epochs, date):
    """Station displacement (m, ECEF) at each epoch (seconds from ``date`` midnight)."""
    station = np.asarray(receiver_xyz, float)
    unit = station/np.linalg.norm(station)
    total = np.zeros((len(np.atleast_1d(epochs)), 3))
    for body, position in (('moon', moon_position(epochs, date)), ('sun', sun_position(epochs, date))):
        distance = np.linalg.norm(position, axis=1)
        direction = position/distance[:, None]
        cosine = direction@unit
        scale = GM_RATIO[body]*R_EARTH**4/distance**3
        total += scale[:, None]*(H2*unit[None, :]*(1.5*cosine**2-0.5)[:, None]
                                 + 3*L2*cosine[:, None]*(direction-cosine[:, None]*unit[None, :]))
    return total


def ocean_loading(block, receiver_xyz, epochs, date):
    """Station displacement (m, ECEF) from a BLQ block (io.blq) at ``epochs``."""
    epochs = np.atleast_1d(np.asarray(epochs, float))
    day = np.floor(epochs/86400)
    fday = epochs-day*86400
    days = (date-datetime(1975, 1, 1)).total_seconds()/86400+day+1.0
    t = (27392.500528+1.000000035*days)/36525.0
    h0 = np.radians(279.69668+36000.768930485*t+3.03e-4*t**2)
    s0 = np.radians(270.434358+481267.88314137*t-0.001133*t**2+1.9e-6*t**3)
    p0 = np.radians(334.329653+4069.0340329577*t-0.010325*t**2-1.2e-5*t**3)
    arguments = np.column_stack([fday, h0, s0, p0, np.full(len(t), 2*np.pi)])@OCEAN_ARGUMENTS.T   # (epochs, 11)
    amplitude, phase = block[:3], np.radians(block[3:])
    up, west, south = (np.sum(amplitude[k]*np.cos(arguments-phase[k]), axis=1) for k in range(3))
    east_axis, north_axis, up_axis = enu_axes(np.asarray(receiver_xyz, float))
    return (-west)[:, None]*east_axis+(-south)[:, None]*north_axis+up[:, None]*up_axis


def mean_pole(year):
    """IERS 2018 secular (linear) mean pole, arcsec."""
    return (55.0+1.677*(year-2000))*1e-3, (320.5+3.460*(year-2000))*1e-3


def pole_tide(receiver_xyz, xp, yp, date):
    """Pole-tide displacement (m, ECEF), IERS 2010 eq. 7.24; ``xp``, ``yp`` in arcsec per epoch."""
    x, y, z = np.asarray(receiver_xyz, float)
    colatitude, longitude = np.pi/2-np.arctan2(z, np.hypot(x, y)), np.arctan2(y, x)
    year = date.year+(date.timetuple().tm_yday-1)/365.25
    xs, ys = mean_pole(year)
    m1, m2 = np.asarray(xp)-xs, -(np.asarray(yp)-ys)
    radial = -33e-3*np.sin(2*colatitude)*(m1*np.cos(longitude)+m2*np.sin(longitude))
    south = -9e-3*np.cos(2*colatitude)*(m1*np.cos(longitude)+m2*np.sin(longitude))
    east = 9e-3*np.cos(colatitude)*(m1*np.sin(longitude)-m2*np.cos(longitude))
    east_axis, north_axis, up_axis = enu_axes(np.asarray(receiver_xyz, float))
    return east[:, None]*east_axis-south[:, None]*north_axis+radial[:, None]*up_axis


def shapiro(satellite_xyz, receiver_xyz):
    """Relativistic signal delay (m) per row."""
    rs = np.linalg.norm(satellite_xyz, axis=1)
    rr = np.linalg.norm(receiver_xyz)
    rho = np.linalg.norm(satellite_xyz-receiver_xyz, axis=1)
    return 2*GM/C**2*np.log((rs+rr+rho)/(rs+rr-rho))


def site_models(combined, satellites, receiver_xyz, date, solid=True, ocean=None, erp=None, relativity=False):
    """Apply the selected station displacements and the Shapiro delay to IF code and phase.

    ``ocean``: BLQ block (io.blq.select_station), ``erp``: io.erp.read_erp output. The total
    displacement dr changes the observed range by -u.dr (u: receiver -> satellite), which is
    removed; the Shapiro delay is removed as well. Columns tide_m / ocean_m / pole_m / shapiro_m
    keep each part (m, added to the modelled range)."""
    from gps_processing.io.erp import pole
    combined = combined.copy()
    receiver_xyz = np.asarray(receiver_xyz, float)
    merged = combined[['epok', 'sat']].merge(satellites[['epok', 'sat', 'x', 'y', 'z']], on=['epok', 'sat'],
                                             how='left', validate='one_to_one')
    xyz = merged[['x', 'y', 'z']].to_numpy(float)
    towards = xyz-receiver_xyz
    towards /= np.linalg.norm(towards, axis=1)[:, None]
    epochs, index = np.unique(combined.epok.to_numpy(float), return_inverse=True)
    parts = {}
    if solid:
        parts['tide_m'] = solid_tide(receiver_xyz, epochs, date)
    if ocean is not None:
        parts['ocean_m'] = ocean_loading(ocean, receiver_xyz, epochs, date)
    if erp is not None:
        parts['pole_m'] = pole_tide(receiver_xyz, *pole(erp, date, epochs), date)
    for column, displacement in parts.items():
        change = np.sum(towards*displacement[index], axis=1)      # observed range is shorter by u.dr
        combined['Pc'] += change
        combined['Lc'] += change
        combined[column] = -change
    if relativity:
        delay = shapiro(xyz, receiver_xyz)
        combined['Pc'] -= delay
        combined['Lc'] -= delay
        combined['shapiro_m'] = delay
    return combined


def apply_solid_tide(combined, satellites, receiver_xyz, date):
    """Remove the tidal range change -u.dr (u: receiver -> satellite) from IF code and phase."""
    return site_models(combined, satellites, receiver_xyz, date)
