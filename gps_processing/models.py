from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from gps_processing.data_directory import require_data_file
from gps_processing.processing.estimation.options import ESTIMATORS

@dataclass
class Observations:
    approximate_xyz: np.ndarray
    raw: pd.DataFrame
    combined: pd.DataFrame
    date: object
    marker: str = ''
    clock_jumps: list = field(default_factory=list)
    antenna_type: str = ''                       # RINEX 'ANT # / TYPE' columns 21-40 (type + radome)
    antenna_delta: tuple = (0.0, 0.0, 0.0)       # 'ANTENNA: DELTA H/E/N' (m)

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
    code_biases: str = ''
    clock_file: str = ''          # RINEX CLK (30 s / 5 s) replacing the SP3 clocks
    antex: str = ''               # ANTEX: satellite PCO, receiver ARP/PCO/PCV
    phase_windup: bool = True     # EKF and PPP-AR only (sequential v2.0 baseline unchanged)
    solid_tides: bool = True      # IERS 2010 solid Earth tide, same scope as the wind-up
    repair_clock_jumps: bool = True
    fcb_files: list = field(default_factory=list)   # reference-station RINEX (>= 5) for FCB estimation
    fcb_method: str = 'all'                         # batch / ekf / ekf+ml / all (best by leave-one-out)
    fcb_interval: float = 900.0                     # narrow-lane block length, s

    ar_products: str = ''
    ar_ratio: float = 3.0
    ar_min_epochs: int = 20
    ar_max_std: float = 0.15

    # Estimation: 'sequential' is the v2.0 baseline; EKF variants add process noise,
    # troposphere, physical corrections and robust/adaptive updates.
    estimator: str = 'ekf'
    dynamics: str = 'static'
    position_noise: float = 10.0
    estimate_troposphere: bool = True
    troposphere_noise: float = 5.0
    tropo_mapping: str = 'niell'      # Niell (1996) or Chao (1972); Saastamoinen ZHD + estimated ZWD
    physical_corrections: bool = True
    robust: bool = True
    code_sigma: float = 0.3
    phase_sigma: float = 0.003
    gf_slip_m: float = 0.05

    integer_method: str = 'lambda'
    ar_success_rate: float = 0.999
    ar_validation: str = 'ratio'
    ml_weighting: bool = False
    ml_threshold: float = 0.99
    ml_validator: str = ''

    reference_xyz: tuple = None

    def validate(self):
        if self.technique not in ('PPP', 'PPP-AR/SDBS', 'PPP-AR/DCM', 'PPP-AR/IPC'):
            raise NotImplementedError('Desteklenen yöntemler: PPP, PPP-AR/SDBS, PPP-AR/DCM, PPP-AR/IPC.')
        if self.estimator not in ESTIMATORS:
            raise ValueError(f'Kestirici {", ".join(ESTIMATORS)} olmalı.')
        from gps_processing.processing.troposphere import MAPPINGS
        if self.tropo_mapping not in MAPPINGS:
            raise ValueError(f'Eşleme fonksiyonu {"/".join(MAPPINGS)} olmalı.')
        if self.dynamics not in ('static', 'kinematic'):
            raise ValueError('Dinamik model static veya kinematic olmalı.')
        from gps_processing.processing.ar.integer import INTEGER_METHODS
        if self.integer_method not in INTEGER_METHODS or self.ar_validation not in ('ratio', 'ml', 'ratio+ml'):
            raise ValueError(f'Tamsayı yöntemi {"/".join(INTEGER_METHODS)}, doğrulama ratio/ml/ratio+ml olmalı.')
        values = [self.position_noise, self.troposphere_noise, self.code_sigma, self.phase_sigma, self.gf_slip_m]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Süreç ve gözlem gürültüleri pozitif olmalı.')
        if not 0 < self.ar_success_rate < 1 or not 0 < self.ml_threshold < 1:
            raise ValueError('Başarı oranı ve ML olasılık eşiği 0 ile 1 arasında olmalı.')
        if self.ml_weighting and self.estimator == 'sequential':
            raise ValueError('ML ağırlıklandırma EKF kestiricisi gerektirir.')
        if self.ml_validator:
            require_data_file(self.ml_validator)
        if self.reference_xyz is not None:
            reference = np.asarray(self.reference_xyz, dtype=float)
            if reference.shape != (3,) or not np.isfinite(reference).all() or np.linalg.norm(reference) < 6e6:
                raise ValueError('Referans koordinat X, Y, Z (metre, ECEF) olmalı.')
        if not np.isfinite([self.ar_ratio,self.ar_max_std]).all() or self.ar_ratio <= 1 or self.ar_max_std <= 0 or self.ar_min_epochs < 2:
            raise ValueError('AR ratio > 1, standart sapma > 0 ve minimum epok >= 2 olmalı.')
        if self.fcb_method not in ('all', 'batch', 'ekf', 'ekf+ml') or not self.fcb_interval > 0:
            raise ValueError('FCB yöntemi all/batch/ekf/ekf+ml, blok süresi pozitif olmalı.')
        network = len(self.fcb_files) >= 5
        if self.technique != 'PPP' and not network and (not self.ar_products or not Path(self.ar_products).is_file()):
            raise ValueError('PPP-AR için AR ürünü gerekli: FCB (*.fcb) / OSB (*.BIA) / JSON ürününü kendiniz seçin '
                             f'ya da FCB kestirimi için en az 5 referans istasyon RINEX dosyası verin (şu an {len(self.fcb_files)}).')
        for extra in [self.ar_products, self.ionosphere, self.troposphere, self.code_biases, self.clock_file, self.antex, *self.fcb_files]:
            if extra:
                require_data_file(extra)
        from gps_processing.pipeline import is_osb
        if self.technique != 'PPP' and is_osb(self.ar_products) and not (self.clock_file and self.antex and self.phase_windup):
            raise ValueError('OSB (Bias-SINEX) AR ürünleri aynı merkezin saat dosyası (RINEX CLK), ANTEX ve faz '
                             'wind-up ile kullanılmalı: faz biasları bu modellerle kestirilmiştir.')
        # fcb_files: reference-station RINEX files for FCB estimation (processing.fcb.network).
        if (self.ionosphere or self.troposphere) and self.technique == 'PPP' and self.estimator == 'sequential':
            raise ValueError('IONEX/SINEX_TRO dosyaları EKF kestiricisiyle kullanılır; Sıralı LS v2.0 temel çözümüdür.')
        if not 0 <= self.elevation_mask < 90:
            raise ValueError('Yükseklik maskesi 0 ile 90 derece arasında olmalı.')
        paths = [Path(p) for p in (self.sp3_before, self.sp3_day, self.sp3_after, self.rinex) if str(p or '').strip() not in ('', '.')]
        for path in paths:
            require_data_file(path)
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError('Önceki, ölçüm ve sonraki gün için farklı SP3 dosyaları seçin (önceki/sonraki gün isteğe bağlı).')

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
    technique: str = 'PPP'
    ar_metadata: dict = field(default_factory=dict)
    residuals: pd.DataFrame = None
    statistics: dict = field(default_factory=dict)
    ml_models: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    output_dir: Path = None      # this process's own folder: <output>/<year>_<doy>_<HHMMSS>
