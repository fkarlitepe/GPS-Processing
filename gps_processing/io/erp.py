"""IGS Earth rotation parameter file (ERP, version 2): polar motion for the pole tide.

Data lines start with the MJD; Xpole and Ypole follow in units of 1e-6 arcsec.
"""
from datetime import datetime
from pathlib import Path
import numpy as np
from gps_processing.io.ionex import read_text

MJD0 = datetime(1858, 11, 17)


def read_erp(path):
    """(mjd, xp, yp) arrays, pole coordinates in arcsec."""
    rows = []
    for line in read_text(path):
        tokens = line.split()
        if len(tokens) < 3:
            continue
        try:
            mjd, x, y = float(tokens[0]), float(tokens[1]), float(tokens[2])
        except ValueError:
            continue
        if 40000 < mjd < 90000:
            rows.append((mjd, x*1e-6, y*1e-6))
    if not rows:
        raise ValueError(f'{Path(path).name}: no ERP data lines (MJD Xpole Ypole).')
    mjd, x, y = np.array(sorted(rows)).T
    return mjd, x, y


def pole(erp, date, epochs):
    """Polar motion (xp, yp) in arcsec at ``epochs`` (s from ``date`` midnight); linear in time,
    held constant up to one day beyond the file."""
    mjd, x, y = erp
    t = (date-MJD0).total_seconds()/86400+np.asarray(epochs, float)/86400
    if t.min() < mjd[0]-1 or t.max() > mjd[-1]+1:
        raise ValueError('ERP file does not cover the observation day.')
    return np.interp(t, mjd, x), np.interp(t, mjd, y)
