"""RINEX 3 and RINEX 2.11 GPS reader. Header observation types determine fixed-width fields.

RINEX 2 types are renamed to their RINEX 3 equivalents so the rest of the processing is shared:
P1 -> C1W, C1 -> C1C, P2 -> C2W, C2 -> C2X, L1/L2/S1/S2 serve both tracking modes (L1W/L1C,
L2W/L2X). With P1 present the code/phase pair is therefore C1W/L1W, otherwise C1C/L1C.
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from gps_processing.constants import C, F1, F2, F5
from gps_processing.models import Observations
from gps_processing.processing.combinations import ionosphere_free, melbourne_wubbena

RINEX2_NAMES = {'P1': ['c1w'], 'C1': ['c1c'], 'L1': ['l1w', 'l1c'], 'S1': ['s1w', 's1c'],
                'P2': ['c2w'], 'C2': ['c2x'], 'L2': ['l2w', 'l2x'], 'S2': ['s2w', 's2x'],
                'C5': ['c5x'], 'L5': ['l5x'], 'S5': ['s5x']}


def read_header(lines):
    version = lines[0][:9].strip() if lines else ''
    types, xyz, system, end, marker = {}, None, None, None, ''
    antenna, delta = '', (0.0, 0.0, 0.0)
    for i, line in enumerate(lines):
        label = line[60:].strip()
        if label == 'MARKER NAME':
            marker = line[:60].strip()
        elif label == 'ANT # / TYPE':
            antenna = line[20:40]
        elif label == 'ANTENNA: DELTA H/E/N':
            delta = tuple(float(v) for v in line[:42].split()[:3])
        elif label == 'APPROX POSITION XYZ':
            xyz = np.array([float(v) for v in line[:60].split()])
        elif label == 'SYS / # / OBS TYPES':
            if line[0] != ' ':
                system = line[0]
                types[system] = []
            types[system].extend(line[7:60].split())
        elif label == '# / TYPES OF OBSERV':
            types.setdefault('G', []).extend(line[6:60].split())
        elif label == 'END OF HEADER':
            end = i + 1
            break
    return version, types, xyz, end, marker, antenna, delta


def rinex3_rows(lines, end, types):
    rows, epoch, first_date, flag = [], None, None, 0
    i = end
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith('>'):
            parts = line[1:].split()
            y, m, d, h, minute = map(int, parts[:5])
            second = float(parts[5])
            flag, count = int(parts[6]), int(parts[7])
            if flag > 1:
                i += count
                epoch = None
                continue
            date = datetime(y, m, d)
            first_date = first_date or date
            epoch = (date-first_date).total_seconds() + h*3600 + minute*60 + second
        elif line.startswith('G') and epoch is not None:
            payload = line[3:]
            while i < len(lines) and lines[i].startswith('   '):
                payload = payload.ljust(((len(payload)+15)//16)*16) + lines[i][3:]
                i += 1
            if not 1 <= int(line[1:3]) <= 32:
                raise ValueError('GPS PRN değeri 1–32 aralığında olmalı.')
            row = {'sat': int(line[1:3]), 'epok': epoch, 'epoch_reset': flag == 1}
            for k, name in enumerate(types['G']):
                cell = payload[k*16:k*16+14].strip()
                row[name.lower()] = float(cell.replace('D', 'E')) if cell else np.nan
                lli = payload[k*16+14:k*16+15].strip()
                if name.startswith('L') and lli and int(lli) & 1:
                    row['epoch_reset'] = True
            rows.append(row)
    return rows, first_date


def rinex2_rows(lines, end, types):
    """RINEX 2.11 body: epoch record (up to 12 satellites per line) and 5 observations per line."""
    names = types['G']
    per_satellite = (len(names)+4)//5
    rows, first_date = [], None
    i = end
    while i < len(lines):
        line = lines[i]
        i += 1
        if len(line) < 29:
            continue
        if not line[:26].strip():
            # Event records (flag 2-5) may leave the epoch blank: skip their header lines.
            if line[26:29].strip().isdigit() and int(line[26:29]) > 1:
                i += int(line[29:32] or 0)
            continue
        flag, count = int(line[26:29] or 0), int(line[29:32] or 0)
        if flag > 1:
            i += count                                 # event: header records follow
            continue
        satellites = line[32:68]
        while len(satellites.replace(' ', '')) < 3*count and i < len(lines):
            satellites += lines[i][32:68]
            i += 1
        ids = [satellites[3*k:3*k+3] for k in range(count)]
        y, m, d, h, minute = (int(v) for v in line[:15].split())
        second = float(line[15:26])
        date = datetime(y+(2000 if y < 80 else 1900), m, d)
        first_date = first_date or date
        epoch = (date-first_date).total_seconds()+h*3600+minute*60+second
        for sat in ids:
            payload = ''.join(lines[i+k][:80].ljust(80) for k in range(per_satellite) if i+k < len(lines))
            i += per_satellite
            system = sat[0] if sat[0] != ' ' else 'G'
            if system != 'G':
                continue
            row = {'sat': int(sat[1:]), 'epok': epoch, 'epoch_reset': flag == 1}
            for k, name in enumerate(names):
                if name not in RINEX2_NAMES:
                    continue
                cell = payload[k*16:k*16+14].strip()
                value = float(cell.replace('D', 'E')) if cell else np.nan
                for target in RINEX2_NAMES[name]:
                    row[target] = value
                lli = payload[k*16+14:k*16+15].strip()
                if name[0] == 'L' and lli.isdigit() and int(lli) & 1:
                    row['epoch_reset'] = True
            rows.append(row)
    return rows, first_date


def read_rinex(path):
    lines = Path(path).read_text(encoding='ascii', errors='replace').splitlines()
    version, types, xyz, end, marker, antenna, delta = read_header(lines)
    if not version[:1] in ('2', '3'):
        raise ValueError('Yalnız RINEX 2.11 ve RINEX 3 gözlem dosyaları destekleniyor.')
    if end is None or xyz is None or xyz.shape != (3,) or not np.isfinite(xyz).all() or np.linalg.norm(xyz) < 1:
        raise ValueError('RINEX başlığı veya yaklaşık XYZ koordinatı eksik/geçersiz.')
    if 'G' not in types:
        raise ValueError('GPS gözlem türleri bulunamadı.')
    rows, first_date = (rinex2_rows if version[0] == '2' else rinex3_rows)(lines, end, types)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError('Dosyada GPS gözlemi bulunamadı.')
    for name in raw.columns:
        if name.startswith('l'):
            frequency = {'1': F1, '2': F2, '5': F5}.get(name[1])
            if frequency:
                raw[name] *= C/frequency
    # Select matching code/phase tracking signals per row, prefer original W signals.
    def pair(band):
        codes = [f'c{band}{s}' for s in ('w', 'c', 'x', 'p', 's', 'l')]
        p, l = pd.Series(np.nan, index=raw.index), pd.Series(np.nan, index=raw.index)
        snr = pd.Series(np.nan, index=raw.index)
        selected = pd.Series('', index=raw.index)
        for name in codes:
            phase = 'l' + name[1:]
            if name in raw and phase in raw:
                valid = p.isna() & raw[name].gt(0) & raw[phase].notna() & raw[phase].ne(0)
                p.loc[valid], l.loc[valid] = raw.loc[valid, name], raw.loc[valid, phase]
                if 's' + name[1:] in raw:
                    snr.loc[valid] = raw.loc[valid, 's' + name[1:]]
                selected.loc[valid] = name.upper() + ' ' + phase.upper()
        return p, l, selected, snr
    p1, l1, signal1, snr1 = pair('1')
    p2, l2, signal2, snr2 = pair('2')
    combined = raw[['sat', 'epok', 'epoch_reset']].copy()
    combined['signals'] = signal1 + ' ' + signal2
    combined['Pc'] = ionosphere_free(p1, p2)
    combined['Lc'] = ionosphere_free(l1, l2)
    combined['A4'] = melbourne_wubbena(p1, p2, l1, l2)
    combined['travel_code'] = p1
    # Geometry-free phase (metres) for cycle-slip checks; SNR (dB-Hz) is an ML feature.
    combined['GF'] = l1 - l2
    combined['snr1'], combined['snr2'] = snr1, snr2
    prior_signals = combined.groupby('sat')['signals'].shift()
    combined['epoch_reset'] |= prior_signals.notna() & combined['signals'].ne(prior_signals)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(subset=['Pc', 'Lc', 'A4'])
    if combined.empty:
        raise ValueError('Geçerli çift frekanslı GPS kod/faz çifti bulunamadı.')
    return Observations(xyz, raw, combined.reset_index(drop=True), first_date, marker,
                        antenna_type=antenna, antenna_delta=delta)
