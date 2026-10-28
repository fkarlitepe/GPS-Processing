"""Shared input location for desktop and command-line processing."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'Data'

def require_data_file(path):
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_relative_to(DATA_DIR.resolve()):
        raise ValueError(f'Girdi dosyası Data klasöründe olmalı: {path}\nGerekli verileri {DATA_DIR} içine koyun.')
    if not candidate.is_file():
        raise ValueError(f'Data içinde girdi dosyası bulunamadı: {path}')
    return candidate
