import tkinter as tk
from tkinter import ttk, filedialog

class FileField(ttk.Frame):
    def __init__(self, parent, label, directory=False):
        super().__init__(parent)
        self.directory = directory
        self.value = tk.StringVar()
        ttk.Label(self, text=label, width=25).pack(side='left')
        ttk.Entry(self, textvariable=self.value).pack(side='left', fill='x', expand=True, padx=8)
        ttk.Button(self, text='Seç…', command=self.browse, width=8).pack(side='left')
    def browse(self):
        path = filedialog.askdirectory() if self.directory else filedialog.askopenfilename()
        if path:
            self.value.set(path)
