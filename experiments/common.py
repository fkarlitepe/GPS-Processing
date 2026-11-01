"""Shared set-up of the MADR experiments (2023-11-23, DOY 327): products, reference coordinate,
session slicing and performance metrics."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from gps_processing.models import ProcessingConfig
from gps_processing.analysis import enu_rotation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'Data'
PRODUCTS = DATA/'products'/'2023327'
OUT = ROOT/'results'/'experiments'
RINEX = DATA/'obs'/'MADR00ESP_R_20233270000_01D_30S_MO.rnx'
SINEX = DATA/'sinex'/'IGS0OPSSNX_20233270000_01D_01D_CRD.SNX'
ATX = DATA/'models'/'igs20_2287.atx'
BLQ = DATA/'models'/'igs_stations_fes.blq'
CENTRES = {          # same-centre sets: orbit, clock, phase/code OSB, attitude, ERP
    'COD': 'COD0MGXFIN', 'GRG': 'GRG0MGXFIN', 'WUM': 'WUM0MGXFIN', 'GFZ': 'GFZ0MGXRAP'}
ERP = {'COD': 'COD0MGXFIN_20233270000_01D_12H_ERP.ERP', 'WUM': 'WUM0MGXFIN_20233270000_01D_01D_ERP.ERP',
       'GFZ': 'GFZ0MGXRAP_20233270000_01D_01D_ERP.ERP', 'GRG': 'COD0MGXFIN_20233270000_01D_12H_ERP.ERP'}
TECHNIQUES = ['PPP', 'PPP-AR/SDBS', 'PPP-AR/DCM', 'PPP-AR/IPC']
SESSION = 7200.0     # s; 12 independent sessions per day


def sinex_xyz(station, path=SINEX):
    values = {}
    for line in open(path):
        if line[7:11] in ('STAX', 'STAY', 'STAZ') and line[14:18] == station:
            values[line[7:11]] = float(line[47:68])
    return np.array([values['STAX'], values['STAY'], values['STAZ']])


def product(centre, kind, doy=327):
    prefix = CENTRES[centre]
    names = {'sp3': '01D_05M_ORB.SP3', 'clk': '01D_30S_CLK.CLK', 'osb': '01D_01D_OSB.BIA', 'att': '01D_30S_ATT.OBX'}
    path = PRODUCTS/centre/f'{prefix}_2023{doy:03d}0000_{names[kind]}'
    return str(path) if path.is_file() else ''


def base_config(centre, technique='PPP', **kw):
    """v2.9 default configuration with the centre's own orbit, clock, OSB and attitude."""
    values = dict(sp3_before=product(centre, 'sp3', 326), sp3_day=product(centre, 'sp3'),
                  sp3_after=product(centre, 'sp3', 328), rinex=str(RINEX), output=str(OUT/'runs'),
                  technique=technique, systems=('G', 'E'), clock_file=product(centre, 'clk'), antex=str(ATX),
                  attitude=product(centre, 'att'), ocean_loading=str(BLQ), erp=str(DATA/'models'/ERP[centre]) if (DATA/'models'/ERP[centre]).is_file() else '',
                  ar_products=product(centre, 'osb') if technique != 'PPP' else '',
                  reference_xyz=tuple(sinex_xyz('MADR')))
    values.update(kw)
    return ProcessingConfig(**values)


# The v2.8 behaviour, used as the baseline of the ablation (no extra models, ratio test).
V28_POLICY = dict(ar_validation='ratio', ar_candidate_gates=False, ar_min_ambiguities=2, ar_max_exclusions=0,
                  ar_min_elevation=0.0, ar_min_arc=0.0)
V28_MODELS = dict(attitude='', satellite_pcv=False, ocean_loading='', erp='', shapiro=False)


def sessions(obs, length=SESSION):
    start = 0.0
    while start < 86400:
        yield start, replace(obs, combined=obs.combined[(obs.combined.epok >= start) & (obs.combined.epok < start+length)].reset_index(drop=True))
        start += length


def enu_errors(positions, reference):
    rotation = enu_rotation(reference)
    d = positions[['x', 'y', 'z']].to_numpy(float)-reference
    return d@rotation.T


def metrics(positions, reference, t0=0.0, horizontal=0.10, vertical=0.10, hold=20, wrong=0.10):
    """Convergence time, time to first fix (TTFF), fix rate, wrong-fix rate and final accuracy."""
    p = positions.sort_values('epok').reset_index(drop=True)
    enu = enu_errors(p, reference)
    h, u = np.hypot(enu[:, 0], enu[:, 1]), np.abs(enu[:, 2])
    t = p.epok.to_numpy(float)-t0
    ok = (h <= horizontal) & (u <= vertical)
    run = np.convolve(ok.astype(int), np.ones(hold, int), 'valid') == hold
    converged = t[np.argmax(run)] if run.any() else np.nan
    fixed = (p.status != 'float').to_numpy() if 'status' in p else np.zeros(len(p), bool)
    frun = np.convolve(fixed.astype(int), np.ones(hold, int), 'valid') == hold
    ttff = t[np.argmax(frun)] if frun.any() else np.nan
    last = t >= t.max()-1800
    d3 = np.linalg.norm(enu, axis=1)
    return dict(convergence_min=converged/60, ttff_min=ttff/60, fix_rate=float(fixed.mean()),
                wrong_fix_rate=float((fixed & (d3 > wrong) & (t > 1800)).sum()/max(fixed.sum(), 1)),
                rms_e=float(np.sqrt(np.mean(enu[last, 0]**2))), rms_n=float(np.sqrt(np.mean(enu[last, 1]**2))),
                rms_u=float(np.sqrt(np.mean(enu[last, 2]**2))), final_3d=float(d3[-1]), epochs=len(p))
