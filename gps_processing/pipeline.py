from gps_processing.io.rinex import read_rinex
from gps_processing.io.sp3 import combine_sp3
from gps_processing.io.export import export_result
from gps_processing.processing.orbit import interpolate_satellites
from gps_processing.processing.ppp import solve
from gps_processing.models import ProcessingResult

def run(config, progress=None):
    progress = progress or (lambda percent, message: None)
    config.validate()
    progress(5, 'SP3 dosyaları okunuyor')
    orbit = combine_sp3(config.sp3_before, config.sp3_day, config.sp3_after)
    progress(20, 'RINEX ve sinyal kombinasyonları')
    obs = read_rinex(config.rinex)
    day = orbit.loc[orbit.epok == 0, 'timestamp']
    if day.empty or day.iloc[0].date() != obs.date.date():
        raise ValueError('RINEX gözlem günü ile ölçüm günü SP3 tarihi eşleşmiyor.')
    progress(40, 'Uydu saatleri ve 9. derece Lagrange enterpolasyonu')
    obs.combined, satellites = interpolate_satellites(orbit, obs.combined)
    progress(65, 'Yükseklik maskesi ve sıralı kod/faz çözümü')
    positions, ambiguities, diagnostics, covariance, corrections = solve(obs.combined, satellites, obs.approximate_xyz, config.elevation_mask)
    result = ProcessingResult(positions, ambiguities, diagnostics, covariance, obs, satellites, orbit, corrections)
    progress(90, 'CSV, Excel ve özet çıktıları yazılıyor')
    export_result(result, config.output)
    progress(100, 'İşlem tamamlandı')
    return result
