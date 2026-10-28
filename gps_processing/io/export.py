from pathlib import Path
import json
import pandas as pd
from gps_processing.processing.geometry import initial_covariance

def export_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        'combine_sp3': result.orbit, 'obs3_gps': result.observations.raw,
        'mw_ion_gps': result.observations.combined,
        'xyz0_apprx': pd.DataFrame(result.observations.approximate_xyz, columns=['coordinate']),
        'result_gps': result.observations.combined, 'sat3_df': result.satellites,
        'cll_result': pd.DataFrame(result.covariance),
        'dx_1': pd.DataFrame(result.corrections, columns=['correction']),
        'n3_amb': result.ambiguities, 'xyz_result': result.positions,
        'diagnostics': result.diagnostics,
    }
    if result.technique == 'PPP' and result.covariance.shape == (36, 36):
        tables['cll'] = pd.DataFrame(initial_covariance())
    if result.technique != 'PPP':
        tables['ar_diagnostics'] = result.diagnostics
    if result.residuals is not None:
        tables['residuals'] = result.residuals
    if result.statistics:
        tables['statistics'] = pd.DataFrame([result.statistics])
    for name, table in tables.items():
        table.to_csv(directory/f'{name}.csv', index=False)
    with pd.ExcelWriter(directory/'results.xlsx', engine='openpyxl') as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
    summary = {'model': result.technique, 'experimental': True, 'epochs_solved': len(result.positions),
               'epochs_skipped': int(result.diagnostics.status.isin(['skipped','yetersiz uydu']).sum()),
               'last_xyz_m': result.positions.iloc[-1][['x','y','z']].tolist(),
               'limitations': ['EKF estimators apply Saastamoinen hydrostatic delay, Chao mapping, estimated ZWD and periodic relativity; the sequential baseline applies none of them.',
                               'Antenna PCO/PCV only with an ANTEX file; phase wind-up uses nominal yaw (no eclipse attitude); no solid-earth/ocean tides. Without a RINEX CLK, SP3 clocks are linearly interpolated (15 min SP3: decimetre clock errors, no N1 fixing).',
                               'AR products: normalized JSON, Bias-SINEX code+phase OSB (with the same centre clocks) or GENFCB L1/L2 FCB; no certified failure-rate control. RINEX 2 is not implemented.',
                               'ML models are research tools: the stochastic model is learned from the same session residuals and the AR validator from simulated ambiguities.']}
    summary['ar_products_and_settings'] = result.ar_metadata
    summary['settings'] = result.settings
    summary['statistics'] = result.statistics
    if 'status' in result.positions:
        summary['solution_status_counts'] = result.positions.status.value_counts().to_dict()
    (directory/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    if result.ml_models:
        (directory/'ml_models.json').write_text(json.dumps(result.ml_models), encoding='utf-8')
    return directory
