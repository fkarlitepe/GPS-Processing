"""Arc bookkeeping: LLI/signal flags, data gaps, Melbourne-Wübbena and geometry-free jumps."""
import numpy as np


class SlipDetector:
    def __init__(self, max_gap=120.0, mw_jump=5.0, gf_jump=None):
        self.max_gap, self.mw_jump, self.gf_jump = max_gap, mw_jump, gf_jump
        self.previous = {}
        self.arc = {}

    def check(self, epoch, frame):
        """Return the PRNs whose arc restarts at this epoch and update the arc lengths."""
        slipped = []
        has_gf = self.gf_jump is not None and 'GF' in frame
        for row in frame.itertuples():
            old = self.previous.get(row.sat)
            reset = old is None or row.epoch_reset or epoch-old[0] > self.max_gap or abs(row.A4-old[1]) > self.mw_jump
            if not reset and has_gf and np.isfinite(old[2]):
                # Ionospheric drift grows with the gap; allow 1 mm/s on top of the base threshold.
                reset = abs(row.GF-old[2]) > self.gf_jump+1e-3*(epoch-old[0])
            if reset:
                slipped.append(int(row.sat))
                self.arc[row.sat] = 0
            self.arc[row.sat] = self.arc.get(row.sat, 0)+1
            self.previous[row.sat] = (epoch, row.A4, row.GF if has_gf else np.nan)
        return slipped
