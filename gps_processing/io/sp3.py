from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from gps_processing import gnss

def read_sp3(path):
    rows, stamp = [], None
    with open(path, encoding='ascii', errors='replace') as source:
        for line in source:
            if line.startswith('*'):
                fields = line[1:].split()
                stamp = datetime(*map(int, fields[:5])) + timedelta(seconds=float(fields[5]))
            elif line[:1] == 'P' and line[1:2] in gnss.SYSTEMS and stamp is not None:
                values = [float(line[a:b]) for a,b in ((4,18),(18,32),(32,46),(46,60))]
                if any(abs(v) >= 999999 for v in values) or np.linalg.norm(values[:3]) == 0:
                    continue
                rows.append((gnss.sat_id(line[1], int(line[2:4])), *values, stamp))
    if not rows:
        raise ValueError(f'Geçerli GPS/Galileo SP3 kaydı yok: {path}')
    return pd.DataFrame(rows, columns=['sat','x','y','z','time','timestamp'])

def combine_sp3(before, day, after):
    """Previous/next day files are optional (empty): a single-day product such as a 5 min MGEX
    orbit is then interpolated one-sided near midnight."""
    frames = [read_sp3(p) if p and str(p) not in ('', '.') else None for p in (before, day, after)]
    dates = [f.timestamp.min().normalize() if f is not None else None for f in frames]
    if (frames[0] is not None and (dates[1]-dates[0]).days != 1) or (frames[2] is not None and (dates[2]-dates[1]).days != 1):
        raise ValueError('SP3 dosyaları sırasıyla önceki gün, ölçüm günü ve sonraki gün olmalı.')
    origin = dates[1]
    frames = [f for f in frames if f is not None]
    combined = pd.concat(frames, ignore_index=True).sort_values(['sat','timestamp'])
    combined = combined.drop_duplicates(['sat','timestamp'])
    combined['epok'] = (combined.timestamp-origin).dt.total_seconds()
    return combined.reset_index(drop=True)
