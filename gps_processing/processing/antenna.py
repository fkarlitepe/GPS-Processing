"""Antenna and attitude models: nominal yaw, satellite PCO, receiver ARP/PCO/PCV, phase wind-up.

Integer-clock (OSB) products are estimated with the IGS ANTEX satellite offsets and with the
phase wind-up applied (Wu et al. 1993), so both must be applied by the user as well; an
unmodelled differential wind-up is a fraction of a narrow-lane cycle and prevents N1 fixing.
Satellite attitude is the nominal yaw unless an ORBEX attitude file (io.orbex) is given, whose
quaternions replace it wherever they cover the observation (eclipse and noon turns included).
The satellite PCV (nadir dependent, ANTEX NOAZI) is applied with the PCO; formulas follow RTKLIB.
"""
from datetime import datetime
import numpy as np
from gps_processing.constants import C, F1, F2
from gps_processing import gnss

LAMBDA_NL = C/(F1+F2)
IF1, IF2 = F1**2/(F1**2-F2**2), -F2**2/(F1**2-F2**2)
# ANTEX frequency codes of the two carriers per system; receivers calibrated for GPS only use
# the GPS values for Galileo (G01 for E1, G02 for E5a), the usual IGS fallback.
ANTEX_FREQUENCIES = {'G': ('G01', 'G02'), 'E': ('E01', 'E05')}
FALLBACK = {'E01': 'G01', 'E05': 'G02'}


def sun_position(epochs, date):
    """Low-precision Sun ECEF position (m), ~0.01 deg (Astronomical Almanac); GPS ≈ UT here."""
    n = (date-datetime(2000, 1, 1, 12)).total_seconds()/86400+np.asarray(epochs, float)/86400
    g = np.radians(357.528+0.9856003*n)
    ecliptic = np.radians(280.460+0.9856474*n+1.915*np.sin(g)+0.020*np.sin(2*g))
    epsilon = np.radians(23.439-4e-7*n)
    distance = (1.00014-0.01671*np.cos(g)-0.00014*np.cos(2*g))*1.495978707e11
    x, y, z = distance*np.cos(ecliptic), distance*np.cos(epsilon)*np.sin(ecliptic), distance*np.sin(epsilon)*np.sin(ecliptic)
    theta = np.radians(280.46061837+360.98564736629*n)
    return np.column_stack([np.cos(theta)*x+np.sin(theta)*y, -np.sin(theta)*x+np.cos(theta)*y, z])


def nominal_axes(satellite, sun):
    """Body x, y, z unit vectors (rows) of a nominally yaw-steered GPS satellite."""
    ez = -satellite/np.linalg.norm(satellite, axis=1)[:, None]
    es = sun-satellite
    es /= np.linalg.norm(es, axis=1)[:, None]
    ey = np.cross(ez, es)
    ey /= np.linalg.norm(ey, axis=1)[:, None]
    ex = np.cross(ey, ez)
    return ex, ey, ez


def enu_axes(xyz):
    x, y, z = xyz
    lat = np.arctan2(z, np.hypot(x, y)*(1-6.69437999014e-3))
    lon = np.arctan2(y, x)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    return east, north, up


def windup(los, ex, ey, receiver_xyz):
    """Fractional wind-up (cycles) per row; ``los`` is the unit vector satellite -> receiver."""
    east, north, _ = enu_axes(receiver_xyz)
    exr, eyr = north, -east
    dot = lambda a, b: np.sum(a*b, axis=1)
    ds = ex-los*dot(los, ex)[:, None]-np.cross(los, ey)
    dr = exr-los*(los@exr)[:, None]+np.cross(los, np.broadcast_to(eyr, los.shape))
    cosine = np.clip(dot(ds, dr)/np.linalg.norm(ds, axis=1)/np.linalg.norm(dr, axis=1), -1, 1)
    phase = np.arccos(cosine)/(2*np.pi)
    return np.where(dot(los, np.cross(ds, dr)) < 0, -phase, phase)


def unwrap(sats, epochs, phase):
    """Continuous wind-up per satellite: phw = ph + floor(previous - ph + 0.5) (RTKLIB), starting
    from previous = 0; vectorized as a period-1 unwrap of each satellite's series."""
    out = phase.copy()
    for sat in np.unique(sats):
        rows = np.flatnonzero(sats == sat)
        rows = rows[np.argsort(epochs[rows], kind='stable')]
        series = np.unwrap(phase[rows], period=1.0)
        out[rows] = series+np.floor(-series[0]+0.5)
    return out


def receiver_correction(antenna, delta_hen, los, receiver_xyz, system='G'):
    """IF and MW antenna corrections (m) to subtract from the observations: -e·(ARP+PCO) + PCV."""
    k = gnss.constants(system)
    east, north, up = enu_axes(receiver_xyz)
    towards = -los                                              # receiver -> satellite
    zenith = np.degrees(np.arccos(np.clip(towards@up, -1, 1)))
    delta = delta_hen[0]*up+delta_hen[1]*east+delta_hen[2]*north
    per_frequency = []
    for key in ANTEX_FREQUENCIES[system]:
        if antenna is not None and key not in antenna.offset:
            key = FALLBACK.get(key, key)
        n, e, u = antenna.offset[key] if antenna is not None else (0.0, 0.0, 0.0)
        offset = delta+n*north+e*east+u*up
        pcv = np.interp(zenith, antenna.zenith, antenna.pcv[key]) if antenna is not None and key in antenna.pcv else 0.0
        per_frequency.append(-towards@offset+pcv)
    d1, d2 = per_frequency
    f1, f2 = k['f1'], k['f2']
    ionosphere_free = k['if1']*d1+k['if2']*d2
    mw = (f1*d1-f2*d2)/(f1-f2)-(f1*d1+f2*d2)/(f1+f2)
    return ionosphere_free, mw


def satellite_pcv(antennas, combined, los, xyz, date):
    """IF and MW satellite PCV (m) per row from the nadir angle (ANTEX NOAZI, nadir grid)."""
    from gps_processing.io.antex import satellite_antenna
    ionosphere_free, mw = np.zeros(len(combined)), np.zeros(len(combined))
    nadir = np.degrees(np.arccos(np.clip(np.sum(-xyz/np.linalg.norm(xyz, axis=1)[:, None]*los, axis=1), -1, 1)))
    sats = combined.sat.to_numpy()
    for sat in np.unique(sats):
        item = satellite_antenna(antennas, int(sat), date)
        system = gnss.system_of(sat)
        one, two = ANTEX_FREQUENCIES[system]
        if item.zenith is None or one not in item.pcv:
            continue
        rows = sats == sat
        d1 = np.interp(nadir[rows], item.zenith, item.pcv[one])
        d2 = np.interp(nadir[rows], item.zenith, item.pcv.get(two, item.pcv[one]))
        k = gnss.constants(system)
        ionosphere_free[rows] = k['if1']*d1+k['if2']*d2
        mw[rows] = (k['f1']*d1-k['f2']*d2)/(k['f1']-k['f2'])-(k['f1']*d1+k['f2']*d2)/(k['f1']+k['f2'])
    return ionosphere_free, mw


def apply_antenna_models(combined, satellites, receiver_xyz, date, antennas=None, receiver_type='',
                         delta_hen=(0.0, 0.0, 0.0), phase_windup=True, attitude=None, pcv=True):
    """Return (combined, satellites, info). ``antennas`` from io.antex.read_antex (None: wind-up only),
    ``attitude`` from io.orbex.read_orbex (None: nominal yaw)."""
    from gps_processing.io.antex import satellite_antenna
    combined, satellites = combined.copy(), satellites.copy()
    receiver_xyz = np.asarray(receiver_xyz, float)
    xyz = satellites[['x', 'y', 'z']].to_numpy(float)
    ex, ey, ez = nominal_axes(xyz, sun_position(satellites.epok.to_numpy(), date))
    info = dict(phase_windup=bool(phase_windup), satellite_pco=False, receiver_antenna='')
    if attitude is not None:
        from gps_processing.io.orbex import axes
        ax, ay, az, covered = axes(attitude, satellites.sat.to_numpy(), satellites.epok.to_numpy(float))
        ex[covered], ey[covered], ez[covered] = ax[covered], ay[covered], az[covered]
        yaw_change = np.degrees(np.arccos(np.clip(np.sum(ax[covered]*nominal_axes(xyz[covered], sun_position(
            satellites.epok.to_numpy()[covered], date))[0], axis=1), -1, 1)))
        info.update(attitude=f'ORBEX quaternions ({covered.mean():.0%} of satellite epochs, nominal yaw elsewhere)',
                    attitude_off_nominal_epochs=int((yaw_change > 5).sum()))
    if antennas is not None:
        offsets = np.zeros_like(xyz)
        for sat in satellites.sat.unique():
            item = satellite_antenna(antennas, int(sat), date)
            k = gnss.constants(gnss.system_of(sat))
            one, two = ANTEX_FREQUENCIES[gnss.system_of(sat)]
            pco = k['if1']*item.offset[one]+k['if2']*item.offset.get(two, item.offset[one])
            rows = (satellites.sat == sat).to_numpy()
            offsets[rows] = ex[rows]*pco[0]+ey[rows]*pco[1]+ez[rows]*pco[2]
        satellites[['x', 'y', 'z']] = xyz+offsets               # centre of mass -> IF phase centre
        xyz = xyz+offsets
        info['satellite_pco'] = True
    merged = combined[['epok', 'sat']].merge(satellites.assign(row=np.arange(len(satellites)))[['epok', 'sat', 'row']],
                                             on=['epok', 'sat'], how='left', validate='one_to_one')
    if merged.row.isna().any():
        raise ValueError('Antenna models: an observation has no satellite position.')
    rows = merged.row.to_numpy(int)
    los = receiver_xyz-xyz[rows]
    los /= np.linalg.norm(los, axis=1)[:, None]
    if antennas is not None and pcv:
        pcv, pcv_mw = satellite_pcv(antennas, combined, los, xyz[rows], date)
        combined['Pc'] -= pcv
        combined['Lc'] -= pcv
        combined['A4'] -= pcv_mw
        info['satellite_pcv'] = True
    cycles = unwrap(combined.sat.to_numpy(), combined.epok.to_numpy(float), windup(los, ex[rows], ey[rows], receiver_xyz))
    if phase_windup:
        combined['windup_cycles'] = cycles
        combined['Lc'] -= gnss.per_row(combined.sat.to_numpy(), 'lambda_n')*cycles    # MW is wind-up free
    if antennas is not None:
        antenna = antennas.get(receiver_type[:20].strip()) or antennas.get((receiver_type[:16].ljust(16)+'NONE').strip())
        correction, mw = np.zeros(len(combined)), np.zeros(len(combined))
        for system in gnss.SYSTEMS:
            rows_s = (combined.sat > 100).to_numpy() if system == 'E' else (combined.sat <= 100).to_numpy()
            if rows_s.any():
                correction[rows_s], mw[rows_s] = receiver_correction(antenna, np.asarray(delta_hen, float), los[rows_s],
                                                                     receiver_xyz, system)
        combined['Pc'] -= correction
        combined['Lc'] -= correction
        combined['A4'] -= mw
        info['receiver_antenna'] = (receiver_type.strip() if antenna is not None else
                                    f'{receiver_type.strip()} not in ANTEX: ARP offset only')
    return combined, satellites, info
