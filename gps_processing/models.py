from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class Observations:
    approximate_xyz: np.ndarray
    raw: pd.DataFrame
    combined: pd.DataFrame
    date: object

@dataclass
class ProcessingConfig:
    sp3_before: Path
    sp3_day: Path
    sp3_after: Path
    rinex: Path
    output: Path = Path('results')
    elevation_mask: float = 10.0
    technique: str = 'PPP'
    ionosphere: str = ''
    troposphere: str = ''
    fcb_files: list = field(default_factory=list)

    def validate(self):
        if self.technique != 'PPP':
            raise NotImplementedError('PPP-AR/SDBS ve Machine Learning özgün projede taslaktı; FCB çözümü henüz uygulanmadı.')
        if self.ionosphere or self.troposphere or self.fcb_files:
            raise NotImplementedError('Harici atmosfer ve FCB dosyası modelleri henüz uygulanmadı. Alanları boş bırakın.')
        if not 0 <= self.elevation_mask < 90:
            raise ValueError('Yükseklik maskesi 0 ile 90 derece arasında olmalı.')
        paths = [Path(p) for p in (self.sp3_before, self.sp3_day, self.sp3_after, self.rinex)]
        for path in paths:
            if not path.is_file():
                raise ValueError(f'Girdi dosyası bulunamadı: {path}')
        if len({p.resolve() for p in paths[:3]}) != 3:
            raise ValueError('Önceki, ölçüm ve sonraki gün için üç farklı SP3 dosyası seçin.')

@dataclass
class ProcessingResult:
    positions: pd.DataFrame
    ambiguities: pd.DataFrame
    diagnostics: pd.DataFrame
    covariance: np.ndarray
    observations: Observations
    satellites: pd.DataFrame
    orbit: pd.DataFrame
    corrections: np.ndarray
