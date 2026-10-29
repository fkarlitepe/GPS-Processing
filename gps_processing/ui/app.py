"""Native desktop UI. Worker threads communicate with Tk only through a queue."""
import queue
import re
import threading
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from gps_processing.models import ProcessingConfig
from gps_processing.data_directory import DATA_DIR, require_data_file
from gps_processing.pipeline import run, compare, slug, VARIANTS, AR_ONLY
from gps_processing.processing.ml.ar_validation import CANDIDATES, DEFAULT_MODEL
from gps_processing.ui.widgets import FileField, LabeledValue, LinePlot

ESTIMATORS = {'EKF (extended Kalman filter)': 'ekf', 'Adaptive EKF (variance factor per observation group)': 'adaptive-ekf', 'Sequential LS (v2.0 baseline)': 'sequential'}
INTEGER = {'LAMBDA + partial AR (PAR)': 'lambda', 'Bounded ILS (v2.0)': 'ils',
           'Integer bootstrapping, decorrelated + PAR': 'bootstrap', 'Cascade rounding (TCAR / CIR)': 'cascade'}
INTEGER_INFO = {
    'lambda': 'Integer least squares after LAMBDA decorrelation; optimal (highest success rate). Partial AR keeps the decorrelated subset that reaches the success-rate threshold.',
    'ils': 'Integer least squares of at most 4 ambiguities, all or nothing (v2.0 baseline).',
    'bootstrap': 'Sequential conditional rounding of the LAMBDA-decorrelated ambiguities: no search, exact success rate, close to ILS (Teunissen et al. 2002, conclusion 4).',
    'cascade': 'TCAR/CIR principle: pre-defined frequency combinations (wide lane, then N1) and each satellite pair rounded on its own, ignoring correlation between pairs. Geometry-free style; lower success rate than LAMBDA.',
}
VALIDATION = {'FFRT — fixed failure rate ratio test': 'ffrt', 'Ratio test + success rate (v2.8)': 'ratio', 'ML validator (bias-aware)': 'ml',
              'FFRT + ML': 'ffrt+ml', 'Ratio + ML': 'ratio+ml'}
TECHNIQUES = {
    'PPP': 'Float PPP: ionosphere-free code + phase; ambiguities stay real-valued. No extra product needed (SP3 + RINEX).',
    'PPP-AR/SDBS': 'Between-satellite single differences (Collins 2008 / Shi 2012 §4.1), WL → N1 fixing. Requires consistent code clocks and satellite WL/N1 bias products.',
    'PPP-AR/DCM': 'Decoupled clock model (Collins 2008, Shi 2012 ch. 3): separate receiver clocks for code, phase and MW. Requires a product with three satellite corrections and a reference-satellite datum.',
    'PPP-AR/IPC': 'Integer phase clock model (Laurichesse et al. 2008): wide lane from the WSB-corrected MW with a satellite-averaged receiver bias, then N1 estimated directly with receiver code and phase clocks. Uses the same three corrections as DCM (code clock, phase clock, WSB).',
}
PLOTS = {
    'ENU error (m)': (['east', 'north', 'up'], ['East', 'North', 'Up'], False),
    '3D / horizontal error (m)': (['error_3d', 'error_h'], ['3D', 'Horizontal'], True),
    'Formal σ (m)': (['sigma_x', 'sigma_y', 'sigma_z'], ['σX', 'σY', 'σZ'], True),
    'Zenith wet delay ZWD (m)': (['zwd_m'], ['ZWD'], False),
    'Mean slant TEC (TECU)': (['stec_tecu'], ['STEC'], False),
    '2nd-order ionosphere, IF phase (mm)': (['iono2_phase_mm'], ['|δ2|'], False),
    'NIS (consistency, expected 1)': (['nis'], ['NIS'], True),
    'Number of satellites': (['satellites', 'gps_satellites', 'galileo_satellites'], ['All', 'GPS', 'Galileo'], False),
    'Code / phase RMS (m)': (['code_rms_m', 'phase_rms_m', 'residual_rms_m'], ['Code', 'Phase', 'AR residual'], True),
}
STAT_COLUMNS = [('solution', 'Method', 190), ('final_3d', 'Final 3D (m)', 90), ('rms_e', 'RMS E', 75), ('rms_n', 'RMS N', 75),
                ('rms_u', 'RMS U', 75), ('rms_3d', 'RMS 3D', 80), ('convergence_s', 'Convergence (min)', 110),
                ('fix_rate', 'Fix rate', 75), ('mean_nis', 'Mean NIS', 75)]
MUTED = '#5d7286'
COMPARE_PLOTS = {'3D error (m)': ('error_3d', True), 'Horizontal error (m)': ('error_h', True), 'Up (m)': ('up', False),
                 'NIS': ('nis', True), 'Formal σZ (m)': ('sigma_z', True)}


def unknowns(technique, ekf, tropo, external_tropo=False, systems=('G',)):
    """Plain-language list of the estimated parameters for the selected setup."""
    both = len(systems) > 1
    zwd = (' · residual zenith wet delay (relative to the SINEX_TRO ZTD)' if external_tropo else ' · zenith wet delay (ZWD)') if tropo else ''
    states = {
        'PPP': f'receiver X, Y, Z · receiver clock offset{zwd} · one float ionosphere-free phase ambiguity per satellite',
        'PPP-AR/SDBS': f'receiver X, Y, Z{zwd} · between-satellite single-differenced WL and N1 ambiguities (receiver clock cancels)',
        'PPP-AR/DCM': f'receiver X, Y, Z · code, phase and MW receiver clocks{zwd} · WL and N1 ambiguities',
        'PPP-AR/IPC': f'receiver X, Y, Z · receiver code and phase clocks{zwd} · N1 ambiguities (WL rounded beforehand with the averaged receiver WL bias)',
    }[technique]
    if both:
        states += (' · Galileo receiver clock (GPS–Galileo time offset and inter-system bias)' if technique in ('PPP', 'PPP-AR/SDBS')
                   else ' · separate Galileo receiver clocks/biases') + ' · one reference satellite per system'
    how = 'prediction + measurement update at every epoch (EKF)' if ekf else 'sequential least squares over the epochs'
    return (f'Estimated unknowns: {states}.\nMethod: {how}. SP3 (satellite positions and clocks), AR and atmosphere products '
            'are inputs treated as known; they are not estimated, only interpolated to the epoch.')


LOGO = Path(__file__).with_name('assets')/'togu_logo.jpg'


def load_logo(height):
    """Tokat Gaziosmanpaşa University logo scaled to ``height`` px; None if Pillow or the file is missing."""
    try:
        from PIL import Image, ImageTk
        image = Image.open(LOGO)
        return ImageTk.PhotoImage(image.resize((round(image.width*height/image.height), height), Image.LANCZOS))
    except (ImportError, OSError):
        return None


def product_kind(name):
    """'ionosphere' for IONEX (*.INX, legacy *.yyI), 'troposphere' for SINEX_TRO (*.TRO, *.zpd), 'code_biases' for Bias-SINEX (*.BSX, *.BIA), 'clock_file' (*.CLK), 'antex' (*.ATX), 'ocean_loading' (*.BLQ), 'erp' (*.ERP); gz allowed."""
    name = re.sub(r'\.(GZ|Z)$', '', name.upper())
    if name.endswith('.INX') or re.search(r'\.\d\dI$', name):
        return 'ionosphere'
    if name.endswith('.TRO') or name.endswith('ZPD'):
        return 'troposphere'
    if name.endswith('.BSX') or name.endswith('.BIA'):
        return 'code_biases'
    if name.endswith('.CLK') or name.endswith('.CLK_30S') or re.search(r'\.\d\dC$', name):
        return 'clock_file'
    if name.endswith('.ATX'):
        return 'antex'
    if name.endswith('.BLQ'):
        return 'ocean_loading'
    if name.endswith('.ERP'):
        return 'erp'
    return None


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('TOGU · GNSS Processing')
        self.geometry('1180x880')
        self.minsize(900, 700)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.events = queue.Queue()
        self.busy = False
        self.result = None
        self.comparison = None
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background='#f3f5f8')
        style.configure('TLabelframe', background='#f3f5f8')
        style.configure('TLabelframe.Label', background='#f3f5f8', foreground='#1f6fb2', font=('DejaVu Sans', 10, 'bold'))
        style.configure('TLabel', background='#f3f5f8', foreground='#172c43', font=('DejaVu Sans',10))
        style.configure('TCheckbutton', background='#f3f5f8')
        style.configure('Title.TLabel', font=('DejaVu Sans',22,'bold'))
        style.configure('Metric.TLabel', font=('DejaVu Sans', 10, 'bold'))
        style.configure('TButton', padding=7)
        style.configure('Start.TButton', padding=(18,10), font=('DejaVu Sans', 11, 'bold'), foreground='#ffffff', background='#1f6fb2')
        style.map('Start.TButton', background=[('disabled', '#9fb3c8'), ('active', '#185a92')])
        shell = ttk.Frame(self, padding=22)
        shell.pack(fill='both', expand=True)
        ttk.Label(shell, text='TOGU / GNSS Processing', style='Title.TLabel').pack(anchor='w')
        ttk.Label(shell, text='Orbits · Observations · EKF estimation · PPP-AR · Machine learning     |     Modular research workspace').pack(anchor='w', pady=(4,16))
        bar = ttk.Frame(shell)
        bar.pack(side='bottom', fill='x', pady=(10,0))              # packed first: never pushed off-screen
        self.button = ttk.Button(bar, text='▶  Start processing  (F5)', command=self.start, style='Start.TButton')
        self.button.pack(side='right', padx=(12,0))
        status_box = ttk.Frame(bar); status_box.pack(side='left', fill='x', expand=True)
        self.status = ttk.Label(status_box, text='Ready — make the selections, then press Start processing'); self.status.pack(anchor='w', pady=(0,4))
        self.progress = ttk.Progressbar(status_box, maximum=100); self.progress.pack(fill='x')
        self.bind('<F5>', lambda event: self.start())
        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill='both', expand=True)
        inputs = ttk.Frame(self.tabs, padding=18)
        settings_outer, settings = self.scrollable(self.tabs)
        output = ttk.Frame(self.tabs, padding=18)
        comparison = ttk.Frame(self.tabs, padding=18)
        about = ttk.Frame(self.tabs, padding=18)
        for panel, name in ((inputs,'01  Data files'),(settings_outer,'02  Processing settings'),(output,'03  Results'),
                            (comparison,'04  Method comparison'),(about,'About')):
            self.tabs.add(panel, text=name)
        self.fields = {}
        self.build_inputs(inputs)
        self.build_settings(settings)
        self.build_output(output)
        self.build_comparison(comparison)
        about_row = ttk.Frame(about); about_row.pack(anchor='nw', fill='x')
        self.logo = load_logo(160)
        if self.logo is not None:
            ttk.Label(about_row, image=self.logo).pack(side='left', anchor='n', padx=(0,24))
        about = ttk.Frame(about_row); about.pack(side='left', anchor='n', fill='x', expand=True)
        ttk.Label(about,text='Tokat Gaziosmanpaşa University\nDr. Furkan KARLITEPE\nfurkan.karlitepe@gop.edu.tr\n\nGNSS Precise Point Positioning (GPS + Galileo) — v2.9\n\n'
                  'Float PPP and PPP-AR user models (SDBS — Ge et al., DCM — Collins, integer phase clock — Laurichesse);\niterated extended Kalman filter '
                  '(process noise, ZWD estimation, IGG-III robust update, adaptive fading), LAMBDA decorrelation and partial AR,\n'
                  'a stochastic model learned from residuals (gradient-boosted trees) and an AR validator trained by Monte Carlo.\n'
                  'External products: IONEX global ionosphere maps (single-layer model, second-order IF term) and\n'
                  'SINEX_TRO station zenith delays (a-priori ZTD + horizontal gradients).\n\n'
                  'Real AR products: RINEX CLK clocks, Bias-SINEX code/phase OSB (integer clocks) and GENFCB FCB; ANTEX PCO/PCV and phase wind-up.\n'
                  'Site models: solid Earth tide, ocean loading (BLQ), pole tide (ERP), Shapiro delay, satellite PCV and ORBEX attitude.\n'
                  'Research software: no geodetic accuracy is guaranteed. Details: docs/PPP_AR.md, docs/EKF_ML.md, docs/ATMOSPHERE.md and docs/MODELS.md.\n\n'
                  'The original sources are preserved in the legacy/ folder.',justify='left',wraplength=950).pack(anchor='w')
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.update_states()
        self.after(100,self.poll)

    def scrollable(self, parent):
        outer = ttk.Frame(parent)
        canvas = tk.Canvas(outer, highlightthickness=0, background='#f3f5f8')
        scroll = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        scroll.pack(side='right', fill='y'); canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(yscrollcommand=scroll.set)
        inner = ttk.Frame(canvas, padding=18)
        window = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.bind_all('<Button-4>', lambda event: canvas.yview_scroll(-2, 'units'))
        canvas.bind_all('<Button-5>', lambda event: canvas.yview_scroll(2, 'units'))
        return outer, inner

    def build_inputs(self, inputs):
        for key,label in [('rinex','RINEX 2.11 / 3 · GNSS observations'),('antex','Antenna model · ANTEX (optional)'),('output','Output folder')]:
            field = FileField(inputs,label,directory=key=='output')
            field.pack(fill='x', pady=10)
            self.fields[key] = field
        self.fields['output'].value.set(str(Path('results').resolve()))
        site = ttk.LabelFrame(inputs, text='Site displacements and signal delays (IERS Conventions 2010, ch. 7)', padding=12)
        site.pack(fill='x', pady=(8, 4))
        for key, label, types in (('ocean_loading', 'Ocean tide loading · BLQ (optional)', [('BLQ ocean loading', '*.blq *.BLQ'), ('All files', '*')]),
                                  ('erp', 'Pole tide · Earth rotation parameters ERP (optional)', [('IGS ERP', '*.erp *.ERP *.erp.gz *.ERP.gz'), ('All files', '*')])):
            field = FileField(site, label, initial=DATA_DIR, filetypes=types)
            field.pack(fill='x', pady=4)
            self.fields[key] = field
        row = ttk.Frame(site); row.pack(fill='x', pady=(6, 2))
        self.shapiro = LabeledValue(row, 'Shapiro (relativistic) delay', 'bool', True, hint='~2 cm, elevation dependent; IGS clocks assume it is applied')
        self.shapiro.pack(fill='x')
        self.satellite_pcv = LabeledValue(row, 'Satellite PCV (ANTEX, nadir)', 'bool', True, hint='needs the ANTEX file above; mm to cm per satellite')
        self.satellite_pcv.pack(fill='x')
        ttk.Label(site, wraplength=950, foreground=MUTED, justify='left', text=
                  'Ocean loading: 11 constituents (M2 … Ssa) for the RINEX marker from a BLQ file of the Onsala loading service '
                  '(holt.oso.chalmers.se/loading, model FES2014b, "displacement in CE frame"); up to several cm near coasts. '
                  'Pole tide: IERS 2010 eq. 7.24 with the polar motion of an IGS ERP file (a few mm). The solid Earth tide is '
                  'switched in Processing settings. Used by the EKF and PPP-AR solutions (not by the v2.0 baseline).').pack(anchor='w', pady=(6, 0))
        ttk.Button(inputs,text='Select files from the Data folder',command=self.load_example).pack(anchor='w',pady=16)
        ttk.Label(inputs,text='Fill the Data folder with the required files before processing.\nThe button selects only the observation file, ANTEX, BLQ, ERP, atmosphere products and the reference stations in Data/network.\nOrbit (SP3), clock (CLK), code-bias and AR products are chosen by you in Processing settings → Positioning technique:\nthe analysis centre you choose changes the solution and the convergence time, so it is never picked automatically.').pack(anchor='w')

    def row(self, rows, widget, **pack):
        """Register a widget for ordered show/hide; ``arrange`` re-packs the visible ones."""
        widget.pack_kwargs = pack or dict(fill='x', pady=2)
        rows.append(widget)
        return widget

    @staticmethod
    def arrange(rows, hidden):
        for widget in rows:
            widget.pack_forget()
        for widget in rows:
            if widget not in hidden:
                widget.pack(**widget.pack_kwargs)

    def build_settings(self, settings):
        self.sections, self.rows = [], {}
        def section(title):
            frame = ttk.LabelFrame(settings, text=title, padding=12)
            frame.title = title
            self.row(self.sections, frame, fill='x', pady=8)
            self.rows[frame] = []
            return frame

        method = self.method_section = section('Positioning technique')
        row = self.row(self.rows[method], ttk.Frame(method), fill='x')
        ttk.Label(row, text='Technique', width=30).pack(side='left')
        self.technique = ttk.Combobox(row,values=list(TECHNIQUES),state='readonly')
        self.technique.set('PPP'); self.technique.pack(side='left',padx=8)
        self.technique.bind('<<ComboboxSelected>>', lambda event: self.update_states())
        self.technique_info = tk.StringVar()
        self.row(self.rows[method], ttk.Label(method, textvariable=self.technique_info, wraplength=950, foreground=MUTED), anchor='w', pady=(6,2))
        row = self.row(self.rows[method], ttk.Frame(method), fill='x', pady=(4,2))
        ttk.Label(row, text='Satellite systems', width=30).pack(side='left')
        self.gps, self.galileo = tk.BooleanVar(value=True), tk.BooleanVar(value=False)
        for text, var in (('GPS (L1/L2)', self.gps), ('Galileo (E1/E5a)', self.galileo)):
            ttk.Checkbutton(row, text=text, variable=var, command=self.update_states).pack(side='left', padx=(8,16))
        self.galileo_hint = self.row(self.rows[method], ttk.Label(method, wraplength=950, foreground='#a4511d', justify='left', text=
                 'Galileo needs: E1 + E5a observations in the RINEX file; a multi-GNSS SP3 and CLK that contain Galileo '
                 '(e.g. *MGX* products — IGS final GPS orbits do not); for PPP-AR an AR product with Galileo biases '
                 '(OSB with E1/E5a, FCB with Galileo records) or at least 5 reference stations tracking Galileo in section 7. '
                 'Each system gets its own receiver clock and reference satellite. The sequential v2.0 baseline is GPS only.'),
                 anchor='w', pady=(0,4))
        orbit_types = [('SP3 orbits', '*.sp3 *.SP3 *.sp3.gz *.SP3.gz *.eph *.EPH'), ('All files', '*')]
        clock_types = [('RINEX clocks', '*.clk *.CLK *.clk_30s *.CLK_30S *.clk.gz *.CLK.gz'), ('All files', '*')]
        bias_types = [('Bias-SINEX', '*.bsx *.BSX *.bia *.BIA *.bsx.gz *.BSX.gz'), ('All files', '*')]
        for key, label, types in (('sp3_day', 'SP3 · observation day', orbit_types),
                                  ('sp3_before', 'SP3 · previous day (optional)', orbit_types),
                                  ('sp3_after', 'SP3 · next day (optional)', orbit_types),
                                  ('clock_file', 'Clocks · RINEX CLK (optional)', clock_types),
                                  ('attitude', 'Attitude · ORBEX quaternions (optional)', [('ORBEX attitude', '*.obx *.OBX *.obx.gz *.OBX.gz'), ('All files', '*')]),
                                  ('code_biases', 'Code biases · Bias-SINEX (optional)', bias_types)):
            self.fields[key] = self.row(self.rows[method], FileField(method, label, initial=DATA_DIR, filetypes=types), fill='x', pady=2)
        self.row(self.rows[method], ttk.Label(method, wraplength=950, foreground=MUTED, justify='left', text=
                 'Orbit and clock products are chosen by you (download them from the analysis centre, e.g. IGS, CODE, GFZ/GBM, CNES/GRG, WHU). '
                 'A 30 s / 5 s CLK replaces the 15 min SP3 clocks; with a single-day SP3 the previous/next day may stay empty. '
                 'The attitude file (*_ATT.OBX) of the same centre replaces the nominal yaw in the PCO and wind-up (eclipse and noon turns). '
                 'Code biases (DSB/OSB) shift codes to the clock datum, GPS C1W/C2W and Galileo C1C/C5Q (Banville et al. 2008); '
                 'not needed with OSB AR products, which carry every signal. The file headers (not only the names) are checked for the analysis centre.'), anchor='w', pady=(0,4))
        self.fields['ar_products'] = self.ar_field = self.row(self.rows[method], FileField(
            method, 'AR product (FCB / OSB / JSON)', initial=DATA_DIR/'fcb',
            filetypes=[('FCB / OSB / JSON AR products', '*.fcb *.FCB *.bia *.BIA *.bsx *.BSX *.json'), ('All files', '*')]),
            fill='x', pady=(6,2))
        self.ar_hint = self.row(self.rows[method], ttk.Label(method, wraplength=950, foreground='#a4511d', justify='left', text=
                 '⚠ Every AR product must be used with the SP3 and CLK of the same analysis centre it was produced with '
                 '(GBM OSB → GBM SP3 + CLK, CODE OSB → COD SP3 + CLK, CNES → GRG SP3 + CLK, an FCB → the orbit/clock it was '
                 'estimated with). Mixing centres gives wrong integer ambiguities. Download the matching files and select them above. '
                 'Left empty, the FCBs are estimated from at least 5 reference stations in section 7.'), anchor='w', pady=(0,4))
        self.mask = self.row(self.rows[method], LabeledValue(method, 'Elevation mask (deg)', 'float', 10, step=1), fill='x', pady=4)
        self.clock_jumps = self.row(self.rows[method], LabeledValue(method, 'Repair receiver clock jumps', 'bool', True,
                                    hint='1 ms jumps common to all satellites are removed from the phase instead of resetting every arc'))

        ekf = section('Estimation — solving for the unknowns')
        rows = self.rows[ekf]
        self.unknowns = tk.StringVar()
        self.row(rows, ttk.Label(ekf, textvariable=self.unknowns, wraplength=950, justify='left'), anchor='w', pady=(0,8))
        self.estimator = self.row(rows, LabeledValue(ekf, 'Estimator', 'choice', list(ESTIMATORS)[0], ESTIMATORS))
        self.dynamics = self.row(rows, LabeledValue(ekf, 'Dynamic model', 'choice', 'static', ['static', 'kinematic'],
                                                    hint='static: fixed site · kinematic: moving receiver'))
        self.position_noise = self.row(rows, LabeledValue(ekf, 'Kinematic position noise', 'float', 10.0, step=0.5, hint='m/√s random walk'))
        self.estimate_troposphere = self.row(rows, LabeledValue(ekf, 'Estimate zenith wet delay (ZWD)', 'bool', True))
        self.troposphere_noise = self.row(rows, LabeledValue(ekf, 'ZWD process noise', 'float', 5.0, step=1, hint='mm/√h random walk'))
        self.physical = self.row(rows, LabeledValue(ekf, 'Physical corrections', 'bool', True, hint='Saastamoinen ZHD + mapping function + periodic relativity'))
        self.mapping = self.row(rows, LabeledValue(ekf, 'Troposphere mapping', 'choice', 'niell', ['niell', 'chao'],
                                                   hint='Niell (1996): latitude, height, day of year · Chao (1972): elevation only'))
        self.windup = self.row(rows, LabeledValue(ekf, 'Phase wind-up', 'bool', True, hint='nominal-yaw wind-up on the IF phase; required by OSB/FCB AR products'))
        self.tides = self.row(rows, LabeledValue(ekf, 'Solid Earth tide', 'bool', True, hint='IERS 2010 degree-2 displacement (up to ~30 cm radial)'))
        self.robust = self.row(rows, LabeledValue(ekf, 'Robust update (IGG-III)', 'bool', True, hint='down-weights outlying observations'))
        self.code_sigma = self.row(rows, LabeledValue(ekf, 'Raw code σ (zenith)', 'float', 0.3, step=0.1, hint='m'))
        self.phase_sigma = self.row(rows, LabeledValue(ekf, 'Raw phase σ (zenith)', 'float', 0.003, step=0.001, hint='m'))
        self.sequential_info = self.row(rows, ttk.Label(ekf, text='Sequential LS reproduces the v2.0 result unchanged: no process noise, troposphere estimation, '
                                                        'physical corrections or atmosphere products. Use it as the baseline in comparisons.',
                                                        wraplength=950, foreground=MUTED), anchor='w', pady=(6,0))
        for item in (self.estimator, self.dynamics):
            item.widget.bind('<<ComboboxSelected>>', self.update_states)
        self.estimate_troposphere.widget.configure(command=self.update_states)

        atmosphere = self.atmosphere_section = section('External atmosphere products')
        rows = self.rows[atmosphere]
        self.fields['ionosphere'] = self.row(rows, FileField(atmosphere, 'Ionosphere map (IONEX)'), fill='x', pady=(0,2))
        self.row(rows, ttk.Label(atmosphere, wraplength=950, foreground=MUTED, justify='left', text=
                 'IGS/CODE global ionosphere map (*.INX, *.INX.gz). Single-layer model at the IONEX shell height: pierce point, '
                 'COSZ slant factor, bilinear VTEC interpolation between Sun-rotated maps. Because the IF combination already '
                 'removes the first-order term, the map is used for the second-order term (7527·c·B·cosθ·STEC, centred-dipole '
                 'field) applied to IF code and phase.'), anchor='w', pady=(0,10))
        self.fields['troposphere'] = self.row(rows, FileField(atmosphere, 'Zenith delay (SINEX_TRO)'), fill='x', pady=(0,2))
        self.row(rows, ttk.Label(atmosphere, wraplength=950, foreground=MUTED, justify='left', text=
                 'IGS station troposphere product (*.TRO, *.TRO.gz, *.zpd) for the RINEX marker or the nearest station within 20 km. '
                 'ZTD is interpolated linearly in time; a-priori ZWD = ZTD − Saastamoinen ZHD with the selected wet mapping, '
                 'gradients with Chen & Herring (1997). If ZWD estimation is on, the filter estimates only the residual to the '
                 'product; otherwise the product delay is applied as fixed.'), anchor='w')
        for key in ('ionosphere', 'troposphere'):
            self.fields[key].value.trace_add('write', lambda *_: self.update_states())

        ar = self.ar_section = section('Integer ambiguity resolution (PPP-AR)')
        rows = self.rows[ar]
        self.integer_method = self.row(rows, LabeledValue(ar, 'Integer method', 'choice', list(INTEGER)[0], INTEGER))
        self.integer_info = tk.StringVar()
        self.row(rows, ttk.Label(ar, textvariable=self.integer_info, wraplength=950, foreground=MUTED), anchor='w', pady=(0,6))
        self.success_rate = self.row(rows, LabeledValue(ar, 'PAR success-rate threshold', 'float', 0.999, step=0.001, hint='subset success rate: exact IB (decorrelated) or IR bound for cascade'))
        self.ar_max_std = self.row(rows, LabeledValue(ar, 'Max. float σ (cycles)', 'float', 0.15, step=0.05, hint='ILS precision gate'))
        self.ar_min_epochs = self.row(rows, LabeledValue(ar, 'Minimum arc length (epochs)', 'int', 20))
        self.ar_min_elevation = self.row(rows, LabeledValue(ar, 'AR elevation cut-off (deg)', 'float', 15.0, step=1, hint='candidates only (PRIDE: 15°)'))
        self.ar_min_arc = self.row(rows, LabeledValue(ar, 'Minimum arc duration (s)', 'float', 900.0, step=60, hint='PRIDE: 15 min'))
        self.ar_min_ambiguities = self.row(rows, LabeledValue(ar, 'Minimum ambiguities per fix', 'int', 4, hint='PRIDE: 4'))
        self.ar_max_exclusions = self.row(rows, LabeledValue(ar, 'Satellite exclusions after a rejection', 'int', 2, hint='lowest Dong & Bock P0 first'))
        self.validation = self.row(rows, LabeledValue(ar, 'Acceptance test', 'choice', list(VALIDATION)[0], VALIDATION))
        self.ar_failure_rate = self.row(rows, LabeledValue(ar, 'FFRT failure rate Pf', 'float', 0.001, step=0.001,
                                                           hint='threshold μ(Pf, Q) simulated per epoch (Verhagen & Teunissen 2013)'))
        self.ar_ratio = self.row(rows, LabeledValue(ar, 'Ratio threshold', 'float', 3.0, step=0.5))
        self.ml_threshold = self.row(rows, LabeledValue(ar, 'ML correct-fix probability threshold', 'float', 0.99, step=0.01,
                                                        hint='calibrated probability; 0.99 ≈ at most 1 % wrong fixes'))
        self.fields['ml_validator'] = self.row(rows, FileField(ar, 'Trained AR validator (.joblib)'), fill='x', pady=6)
        self.ml_model = self.row(rows, LabeledValue(ar, 'Validator model to train', 'choice', DEFAULT_MODEL, CANDIDATES,
                                                    hint='default = best in the fixed-failure-rate benchmark'))
        self.validator_buttons = self.row(rows, ttk.Frame(ar), fill='x')
        ttk.Button(self.validator_buttons, text='Train validator (Monte Carlo) and save to Data/models', command=self.train_validator).pack(side='left')
        ttk.Label(self.validator_buttons, text=f'  If no file is selected, {DEFAULT_MODEL} is trained in memory on first use.', foreground=MUTED).pack(side='left')
        self.ar_hold = self.row(rows, LabeledValue(ar, 'Fix-and-hold', 'bool', False,
                                                   hint='integers fixed identically for N epochs constrain the float filter (PRIDE, RTKLIB)'))
        self.ar_hold_epochs = self.row(rows, LabeledValue(ar, 'Hold after (epochs)', 'int', 10))
        self.ar_fixability = self.row(rows, LabeledValue(ar, 'ML fixability (self-supervised)', 'bool', False,
                                                         hint='float pass → hindsight labels → candidate order, gate and search trigger'))
        diagnostic = self.row(rows, ttk.Frame(ar), fill='x', pady=(8, 0))
        ttk.Button(diagnostic, text='Narrow-lane diagnostic', command=self.start_diagnostic).pack(side='left')
        ttk.Label(diagnostic, wraplength=760, foreground=MUTED, justify='left', text=
                  '  Holds the coordinates (reference above, else the converged float PPP), fixes nothing and reports how close the '
                  'float N1 ambiguities are to integers, with and without the extra models. Uniform fractions = products from '
                  'different centres or a model error; clustered fractions = the products are usable.').pack(side='left')
        self.integer_method.widget.bind('<<ComboboxSelected>>', lambda event: self.update_states())
        self.validation.widget.bind('<<ComboboxSelected>>', lambda event: self.update_states())
        self.ar_hold.widget.configure(command=self.update_states)

        ml = self.ml_section = section('Machine learning — stochastic model')
        self.ml_weighting = self.row(self.rows[ml], LabeledValue(ml, 'ML stochastic model', 'bool', False, hint='GBRT: elevation, SNR, arc length → observation variance'))
        self.row(self.rows[ml], ttk.Label(ml, text='A float EKF pass is run first, observation variances are learned from its residuals and the solution is repeated with the learned weights.',
                                           wraplength=950, foreground=MUTED), anchor='w', pady=(6,0))
        self.ml_weighting.widget.configure(command=self.update_states)

        ref = section('Validation reference')
        row = self.row(self.rows[ref], ttk.Frame(ref), fill='x')
        ttk.Label(row, text='Reference X Y Z (m, ECEF)', width=30).pack(side='left')
        self.reference = tk.StringVar()
        ttk.Entry(row, textvariable=self.reference, width=48).pack(side='left', padx=8)
        ttk.Button(row, text='Take from RINEX header', command=self.reference_from_header).pack(side='left')
        self.row(self.rows[ref], ttk.Label(ref, text='If empty, errors are computed against the last-epoch solution (internal consistency). Enter an IGS/SINEX coordinate to obtain true accuracy and convergence time.',
                                            wraplength=950, foreground=MUTED), anchor='w', pady=(6,0))

        extra = self.fcb_section = section('FCB estimation from a reference network (Li & Zhang 2012)')
        self.row(self.rows[extra], ttk.Label(extra, wraplength=950, foreground=MUTED, justify='left', text=
                 'At least 5 station RINEX files of the same day. Each station is processed with the float EKF PPP using the SP3/CLK/ANTEX '
                 'selected above; wide-lane UFOs come from arc-averaged MW, narrow-lane UFOs from the float IF ambiguities per block. '
                 'Estimators: batch (Li & Zhang integrated adjustment per block), EKF (random-walk UFOs, integers fixed against the '
                 'prediction) and EKF + ML (ambiguity weights learned by gradient boosting). Each is validated by leaving each station out; '
                 'results go to <output>/fcb/ as GENFCB files. A PPP-AR run without an AR product file estimates the FCBs from '
                 'these stations first and uses the best method; with an FCB/OSB file loaded above, that file is used instead. '
                 'Files in Data/network are listed automatically.'), anchor='w', pady=(0,6))
        self.fcb = self.row(self.rows[extra], tk.Listbox(extra,height=5), fill='x')
        buttons = self.row(self.rows[extra], ttk.Frame(extra), fill='x')
        ttk.Button(buttons,text='Add stations',command=self.add_fcb).pack(side='left')
        ttk.Button(buttons,text='Remove selected',command=self.remove_fcb).pack(side='left', padx=6)
        self.fcb_method = self.row(self.rows[extra], LabeledValue(extra, 'FCB estimator', 'choice', 'all', ['all', 'batch', 'ekf', 'ekf+ml']))
        self.fcb_interval = self.row(self.rows[extra], LabeledValue(extra, 'Narrow-lane block', 'float', 900.0, step=300, hint='s (15 min default)'))
        self.fcb_hold = self.row(self.rows[extra], LabeledValue(extra, 'Hold station coordinates', 'bool', True,
                                                                hint='second float pass at the daily coordinate (network practice)'))
        self.fcb_min_contributors = self.row(self.rows[extra], LabeledValue(extra, 'Stations per satellite FCB', 'int', 0,
                                                                            hint='0 = stations/10 (1…10); blocks seen by fewer stations are dropped'))
        self.fcb_workers = self.row(self.rows[extra], LabeledValue(extra, 'Parallel processes', 'int', 0, hint='0 = number of CPUs (max 8); 100+ stations'))
        self.fcb_button = self.row(self.rows[extra], ttk.Button(extra, text='Estimate FCB', command=self.start_fcb), anchor='w', pady=(6,0))

    def build_output(self, output):
        self.summary = ttk.Label(output,text='No processing run yet.'); self.summary.pack(anchor='w',pady=(0,6))
        self.metrics = ttk.Label(output, text='', style='Metric.TLabel'); self.metrics.pack(anchor='w', pady=(0,8))
        inner = ttk.Notebook(output); inner.pack(fill='both', expand=True)
        graph = ttk.Frame(inner, padding=8); table = ttk.Frame(inner, padding=8); diag = ttk.Frame(inner, padding=8)
        inner.add(graph, text='Plot'); inner.add(table, text='Position table'); inner.add(diag, text='Diagnostics / AR')
        row = ttk.Frame(graph); row.pack(fill='x', pady=(0,6))
        ttk.Label(row, text='Quantity').pack(side='left')
        self.plot_choice = ttk.Combobox(row, values=list(PLOTS), state='readonly', width=32)
        self.plot_choice.set(list(PLOTS)[0]); self.plot_choice.pack(side='left', padx=8)
        self.plot_choice.bind('<<ComboboxSelected>>', lambda event: self.show_plot())
        self.plot = LinePlot(graph, height=360); self.plot.pack(fill='both', expand=True)
        columns=('epok','x','y','z','east','north','up','satellites','status')
        self.table = self.treeview(table, columns, 110)
        self.diagnostics = self.treeview(diag, ('epok','status','reason','satellites','wl_ratio','n1_ratio','wl_success','wl_success_upper','n1_success','n1_success_upper','fixed_pairs','nis'), 100)
        self.log=tk.Text(output,height=5,bg='#172c43',fg='#dcebf7',font=('DejaVu Sans Mono',9),state='disabled')
        self.log.pack(fill='x',pady=(10,0))

    def build_comparison(self, panel):
        ttk.Label(panel, text='Runs the selected methods side by side on the same data; each one is written to comparison/<method> in the output folder.\n'
                  'AR variants are shown only when a PPP-AR technique is selected. Errors refer to the reference coordinate if given, otherwise to the last epoch of the first method.',
                  wraplength=1000).pack(anchor='w')
        choices = ttk.Frame(panel); choices.pack(fill='x', pady=8)
        self.variant_vars, self.variant_boxes = {}, {}
        for i, name in enumerate(VARIANTS):
            var = tk.BooleanVar(value=name in ('Sequential LS (v2.0)', 'EKF', 'Adaptive + robust EKF', 'EKF + ML weighting'))
            box = ttk.Checkbutton(choices, text=name, variable=var)
            box.grid(row=i//4, column=i%4, sticky='w', padx=(0,24), pady=2)
            self.variant_vars[name], self.variant_boxes[name] = var, box
        row = ttk.Frame(panel); row.pack(fill='x', pady=(0,8))
        self.compare_button = ttk.Button(row, text='Start comparison →', command=self.start_comparison)
        self.compare_button.pack(side='left')
        ttk.Label(row, text='Plot').pack(side='left', padx=(24,6))
        self.compare_choice = ttk.Combobox(row, values=list(COMPARE_PLOTS), state='readonly', width=20)
        self.compare_choice.set(list(COMPARE_PLOTS)[0]); self.compare_choice.pack(side='left')
        self.compare_choice.bind('<<ComboboxSelected>>', lambda event: self.show_comparison_plot())
        self.stats = ttk.Treeview(panel, columns=[c[0] for c in STAT_COLUMNS], show='headings', height=6)
        for key, title, width in STAT_COLUMNS:
            self.stats.heading(key, text=title); self.stats.column(key, width=width, stretch=True, anchor='e' if key != 'solution' else 'w')
        self.stats.pack(fill='x')
        self.compare_plot = LinePlot(panel, height=330); self.compare_plot.pack(fill='both', expand=True, pady=(8,0))

    def treeview(self, parent, columns, width):
        tree = ttk.Treeview(parent, columns=columns, show='headings', height=12)
        for col in columns:
            tree.heading(col, text=col); tree.column(col, width=width, stretch=True)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=tree.yview); scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(fill='both', expand=True)
        return tree

    def update_states(self, *_):
        technique = self.technique.get()
        ar = technique != 'PPP'
        ekf = ESTIMATORS[self.estimator.get()] != 'sequential'
        method = INTEGER[self.integer_method.get()]
        partial = method != 'ils'
        self.integer_info.set(INTEGER_INFO[method])
        validation = VALIDATION[self.validation.get()]
        if not ekf:
            self.ml_weighting.set(False)
        tropo = ekf and self.estimate_troposphere.get()
        self.technique_info.set(TECHNIQUES[technique])
        self.unknowns.set(unknowns(technique, ekf, tropo, bool(self.fields['troposphere'].value.get().strip()), self.systems()))
        hidden = set()
        if not self.galileo.get():
            hidden.add(self.galileo_hint)
        if not ar:
            hidden |= {self.ar_field, self.ar_hint, self.ar_section, self.fcb_section}     # FCBs only matter for PPP-AR
        if not ekf:
            hidden |= {self.dynamics, self.estimate_troposphere, self.physical, self.mapping, self.windup, self.tides, self.robust, self.ml_section}
            if not ar:
                # v2.0 LS uses its own fixed weights and no atmosphere models.
                hidden |= {self.code_sigma, self.phase_sigma, self.atmosphere_section}
        else:
            hidden.add(self.sequential_info)
        if not (ekf and self.dynamics.get() == 'kinematic'):
            hidden.add(self.position_noise)
        if not tropo:
            hidden.add(self.troposphere_noise)
        hidden.add(self.ar_max_std if partial else self.success_rate)
        if 'ratio' not in validation:
            hidden.add(self.ar_ratio)
        if 'ffrt' not in validation:
            hidden.add(self.ar_failure_rate)
        if not self.ar_hold.get():
            hidden.add(self.ar_hold_epochs)
        if 'ml' not in validation:
            hidden |= {self.ml_threshold, self.fields['ml_validator'], self.ml_model, self.validator_buttons}
        self.arrange(self.sections, hidden)
        visible = [frame for frame in self.sections if frame not in hidden]
        for number, frame in enumerate(visible, 1):
            frame.configure(text=f'{number} · {frame.title}')
        for rows in self.rows.values():
            self.arrange(rows, hidden)
        for name, box in getattr(self, 'variant_boxes', {}).items():
            if name in AR_ONLY:
                box.grid() if ar else box.grid_remove()

    def visible_sections(self):
        return [frame for frame in self.sections if frame.winfo_manager()]

    def reference_from_header(self):
        try:
            from gps_processing.io.rinex import read_rinex
            xyz = read_rinex(require_data_file(self.fields['rinex'].value.get())).approximate_xyz
            self.reference.set(' '.join(f'{v:.4f}' for v in xyz))
        except (ValueError, OSError) as exc:
            messagebox.showerror('RINEX', str(exc))

    def load_example(self):
        if self.busy: return
        for key, field in self.fields.items():
            if key != 'output': field.value.set('')
        self.fcb.delete(0, 'end')
        files=[p for p in DATA_DIR.rglob('*') if p.is_file() and 'models' not in p.parts]
        # Navigation files (*_MN.rnx, *.23n) are not observations.
        is_observation = lambda p: ((p.suffix.lower() in ('.rnx','.obs') or re.fullmatch(r'\.(\d\do|o)', p.suffix.lower()))
                                    and not re.search(r'_[A-Z]N\.RNX$', p.name.upper()))
        # Data/network holds the reference stations for FCB estimation, not the user station.
        network=sorted(p for p in files if is_observation(p) and 'network' in p.relative_to(DATA_DIR).parts)
        observations=[p for p in files if is_observation(p) and p not in network]
        if len(observations)!=1:
            messagebox.showinfo('Data folder', f'{DATA_DIR}\n\nFound {len(observations)} observation files outside Data/network.\nPlace exactly one user observation file in Data (reference stations go to Data/network), or use Browse.')
            return
        try:
            self.fields['rinex'].value.set(str(require_data_file(observations[0])))
            for path in network:
                self.fcb.insert('end', str(require_data_file(path)))
            # Orbit, clock, code-bias and AR products are the user's choice (Positioning technique section).
            # Station / day files (not analysis-centre products): picked when exactly one is present.
            for key in ('ionosphere', 'troposphere', 'antex', 'ocean_loading', 'erp'):
                found = [p for p in files if product_kind(p.name) == key]
                if len(found) == 1:
                    self.fields[key].value.set(str(require_data_file(found[0])))
        except (ValueError, OSError) as exc:
            messagebox.showerror('Data files', str(exc))

    def add_fcb(self):
        from tkinter.filedialog import askopenfilenames
        for path in askopenfilenames(initialdir=str(DATA_DIR)):
            try:
                self.fcb.insert('end', str(require_data_file(path)))
            except ValueError as exc:
                messagebox.showerror('Data folder', str(exc))

    def remove_fcb(self):
        for index in reversed(self.fcb.curselection()): self.fcb.delete(index)

    def systems(self):
        return tuple(s for s, var in (('G', self.gps), ('E', self.galileo)) if var.get())

    def build_config(self):
        values={k:field.value.get().strip() for k,field in self.fields.items()}
        if not self.systems():
            raise ValueError('Select at least one satellite system (GPS and/or Galileo) in Positioning technique.')
        if self.technique.get() != 'PPP' and not values['ar_products'] and self.fcb.size() < 5:
            raise ValueError('PPP-AR needs ambiguity-resolution products.\n\n'
                             '• Section 1 (Positioning technique): select an FCB / OSB product with Browse (Data/fcb), or\n'
                             f'• Section 7 (FCB estimation): add at least 5 reference-station RINEX files so the FCBs are estimated '
                             f'(currently {self.fcb.size()}).')
        if not values['output']: raise ValueError('Select an output folder.')
        if self.atmosphere_section not in self.visible_sections():
            values['ionosphere'] = values['troposphere'] = ''
        reference = self.reference.get().replace(',', ' ').split()
        if reference and len(reference) != 3:
            raise ValueError('Enter X Y Z for the reference coordinate.')
        config=ProcessingConfig(**values,elevation_mask=self.mask.get(),technique=self.technique.get(),fcb_files=list(self.fcb.get(0,'end')), systems=self.systems(),
            fcb_method=self.fcb_method.get(), fcb_interval=self.fcb_interval.get(),
            estimator=ESTIMATORS[self.estimator.get()], dynamics=self.dynamics.get(), position_noise=self.position_noise.get(),
            estimate_troposphere=self.estimate_troposphere.get(), troposphere_noise=self.troposphere_noise.get(),
            physical_corrections=self.physical.get(), robust=self.robust.get(), code_sigma=self.code_sigma.get(),
            phase_sigma=self.phase_sigma.get(), integer_method=INTEGER[self.integer_method.get()],
            ar_success_rate=self.success_rate.get(), ar_validation=VALIDATION[self.validation.get()],
            tropo_mapping=self.mapping.get(), repair_clock_jumps=self.clock_jumps.get(), phase_windup=self.windup.get(), solid_tides=self.tides.get(), ar_ratio=self.ar_ratio.get(), ar_min_epochs=self.ar_min_epochs.get(), ar_max_std=self.ar_max_std.get(),
            ml_weighting=self.ml_weighting.get(), ml_threshold=self.ml_threshold.get(),
            shapiro=self.shapiro.get(), satellite_pcv=self.satellite_pcv.get(), ar_failure_rate=self.ar_failure_rate.get(),
            ar_min_ambiguities=self.ar_min_ambiguities.get(), ar_max_exclusions=self.ar_max_exclusions.get(),
            ar_min_elevation=self.ar_min_elevation.get(), ar_min_arc=self.ar_min_arc.get(), ar_hold=self.ar_hold.get(),
            ar_hold_epochs=self.ar_hold_epochs.get(), ar_fixability=self.ar_fixability.get(), fcb_hold_coordinates=self.fcb_hold.get(),
            fcb_min_contributors=self.fcb_min_contributors.get(), fcb_workers=self.fcb_workers.get(),
            reference_xyz=tuple(float(v) for v in reference) if reference else None)
        config.validate()
        return config

    def launch(self, job, tab):
        if self.busy: return
        try:
            config = self.build_config()
        except (ValueError,NotImplementedError,tk.TclError) as exc:
            messagebox.showerror('Check the inputs',str(exc)); return
        self.busy=True
        for button in (self.button, self.compare_button): button.configure(state='disabled')
        self.progress['value']=0
        self.tabs.select(tab)
        def worker():
            try:
                self.events.put(job(config, lambda p,m:self.events.put(('progress',p,m))))
            except Exception as exc:
                self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def products_confirmed(self):
        """Errors stop the run; an unverifiable FCB/orbit/clock pairing asks the user to confirm."""
        from gps_processing.pipeline import product_consistency
        try:
            errors, warnings = product_consistency(self.build_config())
        except (ValueError, NotImplementedError, tk.TclError):
            return True                                  # launch() reports input errors itself
        if errors:
            messagebox.showerror('Products from different centres', '\n\n'.join(errors)); return False
        if warnings:
            return messagebox.askyesno('Check the orbit / clock products', '\n\n'.join(warnings)+'\n\nContinue anyway?')
        return True

    def start(self):
        if not self.products_confirmed(): return
        self.launch(lambda config, progress: (lambda result: ('done', result, result.output_dir))(run(config, progress)), 2)

    def start_comparison(self):
        if not self.products_confirmed(): return
        names = [name for name, var in self.variant_vars.items() if var.get()]
        self.launch(lambda config, progress: (lambda out: ('compared', *out, out[0].attrs['output']))(compare(config, names, progress)), 3)

    def start_fcb(self):
        stations = list(self.fcb.get(0, 'end'))
        if len(stations) < 5:
            messagebox.showerror('FCB estimation', f'Add at least 5 station RINEX files ({len(stations)} selected).'); return
        from gps_processing.processing.fcb.core import METHODS
        def job(config, progress):
            from dataclasses import replace
            from gps_processing.processing.fcb.network import estimate_fcb
            methods = METHODS if config.fcb_method == 'all' else (config.fcb_method,)
            from gps_processing.pipeline import run_directory
            config = replace(config, technique='PPP', ar_products='', output=run_directory(config.output))
            return ('fcb', estimate_fcb(config, stations, methods, config.fcb_interval, progress=progress))
        self.launch(job, 1)

    def start_diagnostic(self):
        if self.technique.get() == 'PPP':
            messagebox.showerror('Narrow-lane diagnostic', 'Select a PPP-AR technique and an AR product first.'); return
        def job(config, progress):
            from gps_processing.processing.diagnostics import run_nl_diagnostic
            return ('nl', run_nl_diagnostic(config, progress=progress))
        self.launch(job, 2)

    def train_validator(self):
        if self.busy: return
        self.busy = True
        self.events.put(('progress', 10, 'Training the AR validator on Monte Carlo data'))
        def worker():
            try:
                from gps_processing.processing.ml.ar_validation import ARValidator
                model = self.ml_model.get()
                validator = ARValidator.train(samples=8000, model=model)
                target = DATA_DIR/'models'/f'ar_validator_{slug(model)}.joblib'
                target.parent.mkdir(parents=True, exist_ok=True)
                validator.save(target)
                self.events.put(('trained', target, validator.info))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def write_log(self, text):
        self.log.configure(state='normal'); self.log.insert('end',text+'\n'); self.log.see('end'); self.log.configure(state='disabled')

    def poll(self):
        try:
            while True:
                event=self.events.get_nowait()
                if event[0]=='progress':
                    self.progress['value']=event[1]; self.status.configure(text=event[2]); self.write_log(event[2])
                    continue
                self.busy=False
                for button in (self.button, self.compare_button): button.configure(state='normal')
                if event[0]=='error':
                    self.status.configure(text='Processing failed'); messagebox.showerror('Processing error',event[1])
                elif event[0]=='trained':
                    self.fields['ml_validator'].value.set(str(event[1]))
                    info = event[2]
                    self.status.configure(text='AR validator trained')
                    messagebox.showinfo('ML validator', f'{event[1]}\n\nModel: {info["model"]}\nSamples: {info["samples"]} (+{info["calibration_samples"]} for calibration)\nCorrect-fix fraction (simulation): {info["correct_fraction"]:.3f}\nAccuracy on the calibration set: {info["training_accuracy"]:.3f}')
                elif event[0]=='fcb':
                    summary = event[1]['summary']
                    lines = [f'{m}: LOO RMS {self.fmt(e.get("loo_rms_cycles"))} cyc, within 0.15 cyc {self.fmt(100*e.get("loo_within_015", np.nan), 0)}%'
                             for m, e in summary['methods'].items()]
                    lines += [f'⚠ {w}' for w in summary.get('warnings', [])]
                    self.status.configure(text='FCB estimation finished')
                    messagebox.showinfo('FCB estimation', f'{event[1]["output"]}\n\nStations: {", ".join(summary["stations"])}\n' + '\n'.join(lines) + f'\n\nBest (leave-one-station-out): {summary["best_method"]}')
                elif event[0]=='nl':
                    report = event[1]
                    lines = [f'{name}: {e.get("samples", 0)} samples, |fraction| median {self.fmt(e.get("median_abs_fraction"))} cyc, '
                             f'within 0.15 cyc {self.fmt(100*e.get("within_015", np.nan), 0)} % (uniform: 30 %)\n   → {e.get("verdict", "")}'
                             for name, e in report['variants'].items()]
                    self.status.configure(text='Narrow-lane diagnostic finished')
                    messagebox.showinfo('Narrow-lane diagnostic', f'{report["output"]}\nHeld coordinate: {report["reference_source"]}\n\n' + '\n\n'.join(lines))
                elif event[0]=='done':
                    self.show_result(event[1], event[2])
                else:
                    self.show_comparison(event[1], event[2], event[3])
        except queue.Empty: pass
        self.after(100,self.poll)

    @staticmethod
    def fmt(value, digits=3):
        try:
            return '—' if not np.isfinite(float(value)) else f'{float(value):.{digits}f}'
        except (TypeError, ValueError):
            return str(value)

    def show_result(self, result, output):
        self.result = result
        stats = result.statistics
        self.summary.configure(text=f'{result.technique} · {result.settings.get("estimator")} · {len(result.positions)} epochs solved · CSV and Excel: {output}')
        convergence = stats.get('convergence_s', np.nan)
        self.metrics.configure(text=f'Reference: {stats.get("reference")}   |   Final 3D error {self.fmt(stats.get("final_3d"))} m   |   '
            f'RMS E/N/U {self.fmt(stats.get("rms_e"))} / {self.fmt(stats.get("rms_n"))} / {self.fmt(stats.get("rms_u"))} m   |   '
            f'Convergence {self.fmt(convergence/60 if np.isfinite(convergence) else np.nan, 1)} min   |   Fix rate {100*stats.get("fix_rate", 0):.0f}%   |   Mean NIS {self.fmt(stats.get("mean_nis"), 2)}')
        for tree, frame in ((self.table, result.positions), (self.diagnostics, result.diagnostics)):
            for item in tree.get_children(): tree.delete(item)
            for _, row in frame.iterrows():
                tree.insert('', 'end', values=[self.fmt(row[c]) if c in row and isinstance(row[c], (int, float, np.number)) and not isinstance(row[c], bool) else row.get(c, '—') for c in tree['columns']])
        self.show_plot()

    def show_plot(self):
        if self.result is None: return
        columns, labels, log = PLOTS[self.plot_choice.get()]
        frame = self.result.positions
        series = [(label, frame.epok, frame[c]) for c, label in zip(columns, labels) if c in frame]
        self.plot.plot(series, self.plot_choice.get(), self.plot_choice.get(), log=log)

    def show_comparison(self, table, results, output):
        self.comparison = results
        for item in self.stats.get_children(): self.stats.delete(item)
        for _, row in table.iterrows():
            values = []
            for key, *_ in STAT_COLUMNS:
                value = row.get(key)
                if key == 'convergence_s': value = value/60 if np.isfinite(value) else np.nan
                if key == 'fix_rate': value = 100*value
                values.append(value if key == 'solution' else self.fmt(value, 1 if key in ('convergence_s', 'fix_rate') else 3))
            self.stats.insert('', 'end', values=values)
        self.status.configure(text=f'Comparison finished · {Path(output)/"comparison"}')
        self.show_comparison_plot()

    def show_comparison_plot(self):
        if not self.comparison: return
        column, log = COMPARE_PLOTS[self.compare_choice.get()]
        series = [(name, r.positions.epok, r.positions[column]) for name, r in self.comparison.items() if column in r.positions]
        self.compare_plot.plot(series, self.compare_choice.get(), self.compare_choice.get(), log=log)

    def close(self):
        if self.busy:
            messagebox.showinfo('Processing in progress','Wait for the run to finish so that the outputs are complete.'); return
        self.destroy()

def main():
    Application().mainloop()
