import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from gps_processing.data_directory import DATA_DIR, require_data_file

PALETTE = ['#1f6fb2', '#d9822b', '#2e9e5b', '#c23b4a', '#7b5ea7', '#8c6d1f', '#3b9aa8']


class Tooltip:
    """Short explanation shown while the mouse is over a widget (or any of its children); hidden when it leaves."""
    def __init__(self, widget, text, delay=400, wrap=440):
        self.widget, self.text, self.delay, self.wrap = widget, text, delay, wrap
        self.window = self.job = None
        self._bind(widget)

    def _bind(self, widget):
        widget.bind('<Enter>', self.schedule, add='+')
        widget.bind('<Leave>', self.leave, add='+')
        widget.bind('<ButtonPress>', self.hide, add='+')
        for child in widget.winfo_children():
            self._bind(child)

    def inside(self):
        try:
            x, y = self.widget.winfo_pointerxy()
            target = self.widget.winfo_containing(x, y)
        except (KeyError, tk.TclError):      # pointer over a popup that Tk does not know (e.g. a combobox list)
            return False
        while target is not None:
            if target is self.widget:
                return True
            target = target.master
        return False

    def schedule(self, event=None):
        if self.window is None and self.job is None:
            self.job = self.widget.after(self.delay, self.show)

    def leave(self, event=None):
        # Moving between the children of a row also produces <Leave>: hide only if the pointer left the row.
        self.widget.after(30, lambda: None if self.inside() else self.hide())

    def show(self):
        self.job = None
        if not self.text or self.window is not None or not self.widget.winfo_ismapped() or not self.inside():
            return
        window = self.window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        try:
            window.attributes('-topmost', True)
        except tk.TclError:
            pass
        tk.Label(window, text=self.text, justify='left', wraplength=self.wrap, background='#fffbe6', foreground='#172c43',
                 relief='solid', borderwidth=1, padx=8, pady=5, font=('DejaVu Sans', 9)).pack()
        window.update_idletasks()
        x, y = self.widget.winfo_pointerxy()
        width, height = window.winfo_reqwidth(), window.winfo_reqheight()
        x = min(x+16, self.widget.winfo_screenwidth()-width-8)
        y = y+20 if y+20+height < self.widget.winfo_screenheight() else y-height-12
        window.wm_geometry(f'+{max(x, 0)}+{max(y, 0)}')

    def hide(self, event=None):
        if self.job is not None:
            self.widget.after_cancel(self.job)
            self.job = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


class FileField(ttk.Frame):
    def __init__(self, parent, label, directory=False, initial=None, filetypes=None):
        super().__init__(parent)
        self.directory, self.initial, self.filetypes = directory, initial, filetypes
        self.value = tk.StringVar()
        ttk.Label(self, text=label, width=25).pack(side='left')
        ttk.Entry(self, textvariable=self.value).pack(side='left', fill='x', expand=True, padx=8)
        ttk.Button(self, text='Browse…', command=self.browse, width=8).pack(side='left')
    def browse(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        start = self.initial if self.initial is not None and self.initial.is_dir() else DATA_DIR
        options = dict(initialdir=str(start), filetypes=self.filetypes) if self.filetypes else dict(initialdir=str(start))
        path = filedialog.askdirectory() if self.directory else filedialog.askopenfilename(**options)
        if path and not self.directory:
            try:
                path = str(require_data_file(path))
            except ValueError as exc:
                messagebox.showerror('Data folder', str(exc))
                return
        if path:
            self.value.set(path)


class LabeledValue(ttk.Frame):
    """Label + Spinbox/Combobox/Checkbutton row; ``get`` returns the typed value."""
    def __init__(self, parent, label, kind='float', value=0, choices=None, width=12, step=None, hint=''):
        super().__init__(parent)
        self.kind = kind
        ttk.Label(self, text=label, width=30).pack(side='left')
        if kind == 'bool':
            self.var = tk.BooleanVar(value=value)
            self.widget = ttk.Checkbutton(self, variable=self.var)
        elif kind == 'choice':
            self.var = tk.StringVar(value=value)
            self.widget = ttk.Combobox(self, textvariable=self.var, values=list(choices), state='readonly', width=width+12)
        else:
            self.var = tk.StringVar(value=str(value))
            self.widget = ttk.Spinbox(self, textvariable=self.var, from_=0, to=1e6, increment=step or 1, width=width)
        self.widget.pack(side='left', padx=8)
        if hint:
            ttk.Label(self, text=hint, foreground='#5d7286').pack(side='left')
    def get(self):
        value = self.var.get()
        if self.kind == 'float':
            return float(value)
        if self.kind == 'int':
            return int(float(value))
        return value
    def set(self, value):
        self.var.set(value)
    def enable(self, enabled):
        state = ('readonly' if self.kind == 'choice' else 'normal') if enabled else 'disabled'
        self.widget.configure(state=state)


class LinePlot(tk.Canvas):
    """Minimal time-series plot on a Tk canvas (no matplotlib dependency)."""
    def __init__(self, parent, height=300, **kw):
        super().__init__(parent, height=height, background='white', highlightthickness=1,
                         highlightbackground='#cfd8e3', **kw)
        self.series, self.title, self.ylabel, self.log = [], '', '', False
        self.bind('<Configure>', lambda event: self.redraw())
    def plot(self, series, title='', ylabel='', log=False):
        """series: [(label, x, y)], x in seconds."""
        self.series, self.title, self.ylabel, self.log = series, title, ylabel, log
        self.redraw()
    @staticmethod
    def _ticks(low, high, count=5):
        span = high-low or 1.0
        step = 10**math.floor(math.log10(span/count))
        for factor in (1, 2, 5, 10):
            if span/(step*factor) <= count:
                step *= factor
                break
        start = math.ceil(low/step)*step
        return [start+i*step for i in range(int((high-start)/step)+1)]
    def redraw(self):
        self.delete('all')
        width, height = self.winfo_width(), self.winfo_height()
        left, right, top, bottom = 70, 20, 28, 42
        self.create_text(left, 14, text=self.title, anchor='w', font=('DejaVu Sans', 10, 'bold'), fill='#172c43')
        points = []
        for label, xs, ys in self.series:
            pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(y)) and (not self.log or float(y) > 0)]
            points.append((label, pairs))
        values = [p for _, pairs in points for p in pairs]
        if not values or width < 150:
            self.create_text(width/2, height/2, text='No data to display', fill='#7a8a99')
            return
        transform = (lambda v: math.log10(v)) if self.log else (lambda v: v)
        x0, x1 = min(p[0] for p in values), max(p[0] for p in values)
        y0, y1 = min(transform(p[1]) for p in values), max(transform(p[1]) for p in values)
        if x1 == x0: x1 = x0+1
        pad = (y1-y0)*0.08 or 0.5
        y0, y1 = y0-pad, y1+pad
        sx = lambda x: left+(x-x0)/(x1-x0)*(width-left-right)
        sy = lambda y: height-bottom-(transform(y)-y0)/(y1-y0)*(height-top-bottom)
        self.create_rectangle(left, top, width-right, height-bottom, outline='#cfd8e3')
        if self.log:
            ticks = [math.log10(m*10**k) for k in range(math.floor(y0), math.ceil(y1)+1) for m in (1, 2, 5)]
            ticks = [t for t in ticks if y0 <= t <= y1]
        else:
            ticks = self._ticks(y0, y1)
        for tick in ticks:
            y = height-bottom-(tick-y0)/(y1-y0)*(height-top-bottom)
            self.create_line(left, y, width-right, y, fill='#eef2f6')
            label = f'{10**tick:.3g}' if self.log else f'{tick:.3g}'
            self.create_text(left-6, y, text=label, anchor='e', font=('DejaVu Sans', 8), fill='#5d7286')
        xticks = [t*60 for t in self._ticks(x0/60, x1/60, 8)]
        for tick in xticks:
            x = sx(tick)
            self.create_line(x, height-bottom, x, height-bottom+4, fill='#5d7286')
            self.create_text(x, height-bottom+14, text=f'{tick/60:g}', font=('DejaVu Sans', 8), fill='#5d7286')
        self.create_text((left+width-right)/2, height-8, text='Time (minutes since midnight)', font=('DejaVu Sans', 8), fill='#5d7286')
        self.create_text(14, (top+height-bottom)/2, text=self.ylabel, angle=90, font=('DejaVu Sans', 8), fill='#5d7286')
        legend_x = left+8
        for i, (label, pairs) in enumerate(points):
            color = PALETTE[i % len(PALETTE)]
            if len(pairs) > 1:
                self.create_line(*[c for x, y in pairs for c in (sx(x), sy(y))], fill=color, width=2)
            for x, y in pairs if len(pairs) < 60 else []:
                self.create_oval(sx(x)-2, sy(y)-2, sx(x)+2, sy(y)+2, fill=color, outline=color)
            self.create_rectangle(legend_x, top+8, legend_x+10, top+18, fill=color, outline=color)
            item = self.create_text(legend_x+14, top+13, text=label, anchor='w', font=('DejaVu Sans', 8), fill='#172c43')
            legend_x = self.bbox(item)[2]+14
