"""RINEX 3 GPS reader. Header observation types determine fixed-width fields."""
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from gps_processing.constants import C, F1, F2, F5
from gps_processing.models import Observations
from gps_processing.processing.combinations import ionosphere_free, melbourne_wubbena

def read_rinex(path):
    lines = Path(path).read_text(encoding='ascii', errors='replace').splitlines()
    if not lines or not lines[0].lstrip().startswith('3.'):
        raise ValueError('Yalnız RINEX 3 gözlem dosyaları destekleniyor.')
    types, xyz, system, end = {}, None, None, None
    for i, line in enumerate(lines):
        label = line[60:].strip()
        if label == 'APPROX POSITION XYZ':
            xyz = np.array([float(v) for v in line[:60].split()])
        elif label == 'SYS / # / OBS TYPES':
            if line[0] != ' ':
                system = line[0]
                types[system] = []
            types[system].extend(line[7:60].split())
        elif label == 'END OF HEADER':
            end = i + 1
            break
    if end is None or xyz is None or xyz.shape != (3,) or not np.isfinite(xyz).all() or np.linalg.norm(xyz) < 1:
        raise ValueError('RINEX başlığı veya yaklaşık XYZ koordinatı eksik/geçersiz.')
    if 'G' not in types:
        raise ValueError('GPS gözlem türleri bulunamadı.')
    rows, epoch, first_date = [], None, None
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
        for name in codes:
            phase = 'l' + name[1:]
            if name in raw and phase in raw:
                valid = p.isna() & raw[name].gt(0) & raw[phase].notna() & raw[phase].ne(0)
                p.loc[valid], l.loc[valid] = raw.loc[valid, name], raw.loc[valid, phase]
        return p, l
    p1, l1 = pair('1')
    p2, l2 = pair('2')
    combined = raw[['sat', 'epok', 'epoch_reset']].copy()
    combined['Pc'] = ionosphere_free(p1, p2)
    combined['Lc'] = ionosphere_free(l1, l2)
    combined['A4'] = melbourne_wubbena(p1, p2, l1, l2)
    combined['travel_code'] = p1
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(subset=['Pc', 'Lc', 'A4'])
    if combined.empty:
        raise ValueError('Geçerli çift frekanslı GPS kod/faz çifti bulunamadı.')
    return Observations(xyz, raw, combined.reset_index(drop=True), first_date)
