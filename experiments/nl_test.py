"""Narrow-lane product test (Section 2.8) for every analysis centre and one mixed set.

    python3 -m experiments.nl_test [rinex]
"""
import json
import sys
from dataclasses import replace
from experiments import common
from gps_processing.processing.diagnostics import run_nl_diagnostic


def main(rinex=None, systems=('G', 'E')):
    out = common.OUT/'nl_test'
    report = {}
    sets = {centre: common.base_config(centre, 'PPP-AR/SDBS') for centre in common.CENTRES}
    # Negative control: CNES phase biases with GFZ orbits and clocks.
    sets['GRG OSB + GFZ orbit/clock'] = replace(sets['GFZ'], ar_products=common.product('GRG', 'osb'))
    for name, config in sets.items():
        config = replace(config, output=str(out), systems=systems, **({'rinex': rinex} if rinex else {}))
        variants = {'all models': {}, 'without new models': dict(common.V28_MODELS)}
        result = run_nl_diagnostic(config, variants)
        report[name] = result
        print(name, {k: {x: v.get(x) for x in ('samples', 'median_abs_fraction', 'within_010', 'within_015')}
                     for k, v in result['variants'].items()}, flush=True)
    (out/'summary.json').write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main(*(sys.argv[1:2]))
