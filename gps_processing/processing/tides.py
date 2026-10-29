"""Solid Earth tide station displacement, IERS Conventions (2010) eq. 7.5, degree 2, step 1.

    dr = sum_{Moon, Sun} (GM_j/GM_E) (R_E^4/R_j^3) [ h2 r (3/2 (Rj.r)^2 - 1/2) + 3 l2 (Rj.r)(Rj - (Rj.r) r) ]

with unit vectors r (station) and Rj (body), h2 = 0.6078, l2 = 0.0847 (no latitude, degree-3 or
frequency-dependent terms: they are at the millimetre level). The full displacement including
the permanent part is applied, i.e. coordinates are in the conventional tide-free system used
by the IGS. Moon: low-precision series of Montenbruck & Gill (2000, sec. 3.3.2); Sun: see
processing.antenna.sun_position. The radial tide reaches ~30 cm and changes by decimetres within
hours, so without it float PPP ambiguities absorb a time-varying range error.
"""
from datetime import datetime
import numpy as np
from gps_processing.processing.antenna import sun_position

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


def apply_solid_tide(combined, satellites, receiver_xyz, date):
    """Remove the tidal range change -u.dr (u: receiver -> satellite) from IF code and phase."""
    combined = combined.copy()
    merged = combined[['epok', 'sat']].merge(satellites[['epok', 'sat', 'x', 'y', 'z']], on=['epok', 'sat'],
                                             how='left', validate='one_to_one')
    towards = merged[['x', 'y', 'z']].to_numpy(float)-np.asarray(receiver_xyz, float)
    towards /= np.linalg.norm(towards, axis=1)[:, None]
    epochs, index = np.unique(combined.epok.to_numpy(float), return_inverse=True)
    displacement = solid_tide(receiver_xyz, epochs, date)[index]
    change = np.sum(towards*displacement, axis=1)        # observed range is shorter by u.dr
    combined['Pc'] += change
    combined['Lc'] += change
    combined['tide_m'] = -change
    return combined
