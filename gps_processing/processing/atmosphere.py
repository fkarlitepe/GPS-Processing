"""Slant atmospheric delays shared by the float PPP and PPP-AR filters.

Troposphere: Saastamoinen ZHD + Chao mapping by default. With a SINEX_TRO file the
station ZTD (linear in time) gives the a-priori wet delay ZWD0 = ZTD - ZHD and the
north/east gradients are mapped with Chen & Herring (1997); the filter's ZWD state then
estimates only the residual to ZWD0.
Ionosphere: with an IONEX file the second-order term of the IF combination is applied
(see ``processing.ionosphere``); the first-order term is already eliminated.
"""
from dataclasses import dataclass
import numpy as np
from gps_processing.analysis import enu_rotation
from gps_processing.processing.troposphere import Troposphere, geodetic
from gps_processing.processing import ionosphere


@dataclass
class SlantDelays:
    common: np.ndarray       # m, added to code and phase: hydrostatic + ZWD0 + gradients
    wet: np.ndarray          # wet mapping factor (partial of the ZWD state)
    zwd_apriori: float       # m
    code: np.ndarray         # m, IF code only (second-order ionosphere)
    phase: np.ndarray        # m, IF phase only
    stec: np.ndarray         # TECU, NaN without IONEX


class Atmosphere:
    def __init__(self, xyz, reference=None, marker='', troposphere_file='', ionosphere_file='', physical=True):
        self.model = Troposphere(xyz)
        self.physical = physical
        self.tro = self.gim = None
        if troposphere_file:
            from gps_processing.io.sinex_tro import read_sinex_tro, select_site
            self.tro = select_site(read_sinex_tro(troposphere_file, reference), marker, np.asarray(xyz, dtype=float))
        if ionosphere_file:
            from gps_processing.io.ionex import read_ionex
            self.gim = read_ionex(ionosphere_file, reference)

    @classmethod
    def from_config(cls, config, observations):
        from gps_processing.processing.estimation import FilterOptions
        return cls(observations.approximate_xyz, observations.date, getattr(observations, 'marker', ''),
                   getattr(config, 'troposphere', ''), getattr(config, 'ionosphere', ''),
                   FilterOptions.from_config(config).physical_corrections)

    @property
    def initial_zwd(self):
        """State start value: residual ZWD (0) with an external ZTD, otherwise the model ZWD."""
        return 0.0 if self.tro is not None else self.model.zwd

    @property
    def initial_zwd_variance(self):
        if self.tro is None:
            return 0.3**2
        sigma = np.nanmedian(self.tro.ztd_sigma) if np.isfinite(self.tro.ztd_sigma).any() else 0.005
        return max(float(sigma), 0.005)**2*4

    def zwd_apriori(self, epoch):
        if self.tro is None:
            return 0.0
        t = self.tro.epochs
        if epoch < t[0]-3600 or epoch > t[-1]+3600:
            raise ValueError(f'SINEX_TRO ({self.tro.site}) does not cover the observation epochs.')
        return float(np.interp(epoch, t, self.tro.ztd))-self.model.zhd

    def slant(self, epoch, receiver, satellites, elevation_deg):
        receiver = np.asarray(receiver, dtype=float)
        n = len(satellites)
        dry, wet = self.model.mapping(elevation_deg)
        e = np.radians(np.maximum(elevation_deg, 1.0))
        enu = (np.asarray(satellites)-receiver)@enu_rotation(receiver).T
        azimuth = np.arctan2(enu[:, 0], enu[:, 1])
        zwd0 = self.zwd_apriori(epoch)
        if self.tro is not None:
            gn = float(np.interp(epoch, self.tro.epochs, self.tro.gradient_north))
            ge = float(np.interp(epoch, self.tro.epochs, self.tro.gradient_east))
            gradient = (gn*np.cos(azimuth)+ge*np.sin(azimuth))/(np.sin(e)*np.tan(e)+0.0032)
            common = self.model.zhd*dry+zwd0*wet+gradient
        else:
            common = self.model.zhd*dry if self.physical else np.zeros(n)
        code, phase, stec = np.zeros(n), np.zeros(n), np.full(n, np.nan)
        if self.gim is not None:
            lat, lon, _ = geodetic(receiver)
            lat_p, lon_p, factor = ionosphere.pierce_point(lat, lon, e, azimuth, self.gim.radius_km, self.gim.height_km)
            stec = ionosphere.vtec(self.gim, epoch, np.degrees(lat_p), np.degrees(lon_p))*factor
            shell = (self.gim.radius_km+self.gim.height_km)*1e3
            ipp = shell*np.c_[np.cos(lat_p)*np.cos(lon_p), np.cos(lat_p)*np.sin(lon_p), np.sin(lat_p)]
            direction = receiver-np.asarray(satellites)
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            valid = np.isfinite(stec)
            code[valid], phase[valid] = ionosphere.ionosphere_free_second_order(
                stec[valid], ionosphere.dipole_field(ipp[valid]), direction[valid])
        return SlantDelays(common, wet, zwd0, code, phase, stec)

    def describe(self):
        tro = (f'SINEX_TRO {self.tro.site} ZTD + gradients (Chen & Herring)' if self.tro is not None
               else 'Saastamoinen ZHD' if self.physical else 'none')
        iono = (f'IONEX single-layer {self.gim.height_km:g} km, second-order IF term (dipole field)'
                if self.gim is not None else 'ionosphere-free combination only')
        return dict(troposphere_model=tro, ionosphere_model=iono)
