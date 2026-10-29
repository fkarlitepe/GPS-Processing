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

def read_config_values(path):
    """Field values stored in a configuration file (lists of systems/coordinates as tuples)."""
    import json
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    suffix = path.suffix.lower()
    if suffix == '.json':
        values = json.loads(text)
    elif suffix == '.toml':
        try:
            import tomllib
        except ImportError:                     # Python 3.10
            try:
                import tomli as tomllib
            except ImportError as exc:
                raise ValueError('TOML configuration needs Python 3.11 or the tomli package; JSON works everywhere.') from exc
        values = tomllib.loads(text)
    elif suffix in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as exc:
            raise ValueError('YAML configuration needs PyYAML (pip install pyyaml); JSON and TOML work without it.') from exc
        values = yaml.safe_load(text) or {}
    else:
        raise ValueError(f'{path.name}: configuration must be .json, .toml or .yaml.')
    unknown = sorted(set(values)-set(ProcessingConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f'{path.name}: unknown configuration keys: {", ".join(unknown)}.')
    for key in ('systems', 'reference_xyz'):
        if isinstance(values.get(key), list):
            values[key] = tuple(values[key])
    return values


@dataclass
class ProcessingConfig:
    sp3_before: Path
    sp3_day: Path
    sp3_after: Path
    rinex: Path
    output: Path = Path('results')
    elevation_mask: float = 10.0
    technique: str = 'PPP'
    systems: tuple = ('G',)           # 'G' GPS (L1/L2), 'E' Galileo (E1/E5a); chosen in the UI / --systems
    ionosphere: str = ''
    troposphere: str = ''
    code_biases: str = ''
    clock_file: str = ''          # RINEX CLK (30 s / 5 s) replacing the SP3 clocks
    antex: str = ''               # ANTEX: satellite PCO, receiver ARP/PCO/PCV
    phase_windup: bool = True     # EKF and PPP-AR only (sequential v2.0 baseline unchanged)
    solid_tides: bool = True      # IERS 2010 solid Earth tide, same scope as the wind-up
    ocean_loading: str = ''       # BLQ ocean tide loading coefficients (Onsala service)
    erp: str = ''                 # IGS ERP file: polar motion for the pole tide
    shapiro: bool = True          # relativistic (Shapiro) signal delay
    satellite_pcv: bool = True    # nadir-dependent satellite PCV from the ANTEX file
    attitude: str = ''            # ORBEX attitude quaternions (*_ATT.OBX) of the orbit/clock centre
    repair_clock_jumps: bool = True
    fcb_files: list = field(default_factory=list)   # reference-station RINEX (>= 5) for FCB estimation
    fcb_method: str = 'all'                         # batch / ekf / ekf+ml / all (best by leave-one-out)
    fcb_interval: float = 900.0                     # narrow-lane block length, s
    fcb_hold_coordinates: bool = True               # second float pass with the daily coordinate held
    fcb_min_contributors: int = 0                   # stations per satellite FCB of a block; 0: stations/10 in [1, 10]
    fcb_workers: int = 0                            # parallel station processes (0: CPU count, max 8)

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
    ar_validation: str = 'ffrt'       # ffrt / ratio / ml / ratio+ml / ffrt+ml (processing.ar.policy)
    ar_failure_rate: float = 0.001    # FFRT target failure rate Pf
    ar_min_ambiguities: int = 4       # PRIDE: at least four ambiguities per fixed subset
    ar_max_exclusions: int = 2        # satellites dropped (lowest P0 first) after a rejected stage
    ar_min_elevation: float = 15.0    # deg, AR candidates only (the float filter uses elevation_mask)
    ar_min_arc: float = 900.0         # s, minimum arc duration of an AR candidate
    ar_candidate_gates: bool = True   # PRIDE fraction (WL 0.20 / N1 0.15 cycle) and sigma (0.25 cycle) gates
    ar_hold: bool = False             # fix-and-hold: feed repeated fixes back to the float filter
    ar_fixability: bool = False       # self-supervised fixability model (hindsight labels, two passes)
    ar_fixability_threshold: float = 0.5
    ar_hold_epochs: int = 10
    ml_weighting: bool = False
    ml_threshold: float = 0.99
    ml_validator: str = ''

    reference_xyz: tuple = None
    hold_coordinates: bool = False    # position held at reference_xyz (NL diagnostics, network stations)

    @classmethod
    def from_file(cls, path, **overrides):
        """Configuration from a JSON, TOML (Python 3.11+ or tomli) or YAML (PyYAML) file; keys are
        the field names, ``overrides`` win (command-line options)."""
        values = {**read_config_values(path), **overrides}
        return cls(**values)

    def to_file(self, path):
        """Write every field as JSON (paths as strings); the file can be edited and reused with --config."""
        import json
        from dataclasses import asdict
        values = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    def validate(self):
        from gps_processing import gnss
        if not self.systems or not set(self.systems) <= set(gnss.SYSTEMS):
            raise ValueError('En az bir uydu sistemi seçin: GPS (G) ve/veya Galileo (E).')
        if 'E' in self.systems and self.technique == 'PPP' and self.estimator == 'sequential':
            raise ValueError('Galileo, EKF kestiricisiyle çözülür; Sıralı LS v2.0 temel çözümü yalnız GPS içindir.')
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
        from gps_processing.processing.ar.policy import VALIDATIONS
        if self.integer_method not in INTEGER_METHODS or self.ar_validation not in VALIDATIONS:
            raise ValueError(f'Tamsayı yöntemi {"/".join(INTEGER_METHODS)}, doğrulama {"/".join(VALIDATIONS)} olmalı.')
        if not 0 < self.ar_failure_rate < 0.1 or self.ar_min_ambiguities < 1 or self.ar_max_exclusions < 0 or self.ar_hold_epochs < 1:
            raise ValueError('FFRT hata oranı 0-0.1, en az 1 belirsizlik, dışlama >= 0 ve tutma epoku >= 1 olmalı.')
        if self.hold_coordinates and (self.reference_xyz is None or self.dynamics != 'static'):
            raise ValueError('Koordinat sabitleme referans koordinat ve statik model gerektirir.')
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
        if self.fcb_method not in ('all', 'batch', 'ekf', 'ekf+ml') or not self.fcb_interval > 0 or self.fcb_min_contributors < 0 or self.fcb_workers < 0:
            raise ValueError('FCB yöntemi all/batch/ekf/ekf+ml, blok süresi pozitif olmalı.')
        network = len(self.fcb_files) >= 5
        if self.technique != 'PPP' and not network and (not self.ar_products or not Path(self.ar_products).is_file()):
            raise ValueError('PPP-AR için AR ürünü gerekli: FCB (*.fcb) / OSB (*.BIA) / JSON ürününü kendiniz seçin '
                             f'ya da FCB kestirimi için en az 5 referans istasyon RINEX dosyası verin (şu an {len(self.fcb_files)}).')
        for extra in [self.ar_products, self.ionosphere, self.troposphere, self.code_biases, self.clock_file, self.antex,
                      self.ocean_loading, self.erp, self.attitude, *self.fcb_files]:
            if extra:
                require_data_file(extra)
        from gps_processing.io.products import is_osb
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
