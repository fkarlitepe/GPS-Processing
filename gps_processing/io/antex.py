"""ANTEX 1.4 reader: GPS satellite PCO (body frame) and receiver PCO/PCV (NOAZI) on L1/L2."""
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from gps_processing.io.ionex import read_text


@dataclass
class Antenna:
    name: str
    serial: str
    valid_from: datetime = datetime.min
    valid_until: datetime = datetime.max
    zenith: np.ndarray = None               # ZEN1..ZEN2 grid (deg)
    offset: dict = field(default_factory=dict)      # 'G01' -> (3,) metres (sat: x, y, z; receiver: N, E, U)
    pcv: dict = field(default_factory=dict)         # 'G01' -> NOAZI values (metres) on ``zenith``


def antex_time(line):
    values = line[:43].split()
    return datetime(*(int(float(v)) for v in values[:5])) if values else None


def read_antex(path, satellites=True, receiver=None):
    """Return {'G05': [Antenna, ...]} for GPS satellites and/or {receiver type: Antenna};
    ``receiver`` is a 20-character ANTEX type+radome or a list of them."""
    receivers = {r.ljust(20)[:20] for r in ([receiver] if isinstance(receiver, str) else receiver or [])}
    out, current, frequency = {}, None, None
    for line in read_text(path):
        label = line[60:].strip()
        if label == 'START OF ANTENNA':
            current = None
        elif label == 'TYPE / SERIAL NO':
            name, serial = line[:20].strip(), line[20:40].strip()
            is_satellite = serial[:1] == 'G' and serial[1:3].isdigit() and len(serial) == 3
            wanted = (satellites and is_satellite) or line[:20] in receivers
            current = Antenna(name, serial) if wanted else None
        elif current is None:
            continue
        elif label == 'ZEN1 / ZEN2 / DZEN':
            z1, z2, dz = (float(v) for v in line[2:20].split())
            current.zenith = np.arange(z1, z2+dz/2, dz)
        elif label == 'VALID FROM':
            current.valid_from = antex_time(line)
        elif label == 'VALID UNTIL':
            current.valid_until = antex_time(line)
        elif label == 'START OF FREQUENCY':
            frequency = line[3:6]
        elif label == 'END OF FREQUENCY':
            frequency = None
        elif label == 'NORTH / EAST / UP' and frequency:     # satellites: x, y, z body frame
            current.offset[frequency] = np.array([float(v) for v in line[:30].split()])*1e-3
        elif frequency and line[3:8] == 'NOAZI':
            current.pcv[frequency] = np.array([float(v) for v in line[8:].split()])*1e-3
        elif label == 'END OF ANTENNA':
            if current.serial[:1] == 'G' and len(current.serial) == 3:
                out.setdefault(current.serial, []).append(current)
            else:
                out[current.name] = current
            current = None
    return out


def satellite_antenna(antennas, prn, date):
    for item in antennas.get(f'G{prn:02d}', []):
        if item.valid_from <= date < item.valid_until:
            return item
    raise ValueError(f'G{prn:02d}: no ANTEX satellite entry valid on {date:%Y-%m-%d}.')
