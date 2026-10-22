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
        'cll': pd.DataFrame(initial_covariance()), 'cll_result': pd.DataFrame(result.covariance),
        'dx_1': pd.DataFrame(result.corrections, columns=['correction']),
        'n3_amb': result.ambiguities, 'xyz_result': result.positions,
        'diagnostics': result.diagnostics,
    }
    for name, table in tables.items():
        table.to_csv(directory/f'{name}.csv', index=False)
    with pd.ExcelWriter(directory/'results.xlsx', engine='openpyxl') as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)
    summary = {'model': 'experimental float PPP', 'epochs_solved': len(result.positions),
               'epochs_skipped': int((result.diagnostics.status != 'çözüldü').sum()),
               'last_xyz_m': result.positions.iloc[-1][['x','y','z']].tolist(),
               'limitations': ['No tropospheric model, antenna corrections, relativistic correction or precise clock product.',
                               'PPP-AR, external atmosphere models and RINEX 2 are not implemented.']}
    (directory/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return directory
