"""Native desktop UI. Worker threads communicate with Tk only through a queue."""
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from gps_processing.models import ProcessingConfig
from gps_processing.pipeline import run
from gps_processing.ui.widgets import FileField

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('TOGU · GPS Processing')
        self.geometry('1080x820')
        self.minsize(850, 680)
        self.events = queue.Queue()
        self.busy = False
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background='#f3f5f8')
        style.configure('TLabel', background='#f3f5f8', foreground='#172c43', font=('DejaVu Sans',10))
        style.configure('Title.TLabel', font=('DejaVu Sans',22,'bold'))
        style.configure('TButton', padding=7)
        shell = ttk.Frame(self, padding=22)
        shell.pack(fill='both', expand=True)
        ttk.Label(shell, text='TOGU / GPS Processing', style='Title.TLabel').pack(anchor='w')
        ttk.Label(shell, text='Yörünge · Gözlem · Konum çözümü     |     Modüler araştırma çalışma alanı').pack(anchor='w', pady=(4,16))
        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill='both', expand=True)
        inputs = ttk.Frame(self.tabs, padding=18)
        settings = ttk.Frame(self.tabs, padding=18)
        output = ttk.Frame(self.tabs, padding=18)
        about = ttk.Frame(self.tabs, padding=18)
        for panel, name in ((inputs,'01  Veri dosyaları'),(settings,'02  İşlem ayarları'),(output,'03  Sonuçlar'),(about,'Hakkında')):
            self.tabs.add(panel, text=name)
        self.fields = {}
        for key,label in [('sp3_before','SP3 · Önceki gün'),('sp3_day','SP3 · Ölçüm günü'),('sp3_after','SP3 · Sonraki gün'),('rinex','RINEX 3 · GPS gözlemi'),('output','Çıktı klasörü')]:
            field = FileField(inputs,label,directory=key=='output')
            field.pack(fill='x', pady=10)
            self.fields[key] = field
        self.fields['output'].value.set(str(Path('results').resolve()))
        ttk.Button(inputs,text='Projedeki örnek verileri yükle',command=self.load_example).pack(anchor='w',pady=16)
        ttk.Label(inputs,text='Üç ardışık günün SP3 verisini seçin. Ölçüm günü RINEX tarihiyle eşleşmelidir.').pack(anchor='w')
        ttk.Label(settings,text='İşlem tekniği').pack(anchor='w')
        self.technique = ttk.Combobox(settings,values=['PPP','PPP-AR/SDBS','PPP-AR/Machine_Learning'],state='readonly')
        self.technique.set('PPP'); self.technique.pack(anchor='w',pady=8)
        row = ttk.Frame(settings); row.pack(fill='x',pady=8)
        ttk.Label(row,text='Yükseklik maskesi (derece)').pack(side='left')
        self.mask = ttk.Spinbox(row,from_=0,to=89,width=8); self.mask.set('10'); self.mask.pack(side='left',padx=12)
        for key,label in [('ionosphere','İyonosfer dosyası'),('troposphere','Troposfer dosyası')]:
            field = FileField(settings,label); field.pack(fill='x',pady=6); self.fields[key]=field
        ttk.Label(settings,text='FCB kestirimi için referans gözlem dosyaları').pack(anchor='w',pady=(12,4))
        self.fcb = tk.Listbox(settings,height=4); self.fcb.pack(fill='x')
        buttons=ttk.Frame(settings); buttons.pack(fill='x')
        ttk.Button(buttons,text='Referans ekle',command=self.add_fcb).pack(side='left')
        ttk.Button(buttons,text='Seçileni kaldır',command=self.remove_fcb).pack(side='left')
        ttk.Label(settings,text='PPP-AR, RINEX 2 ve harici atmosfer modelleri özgün projede taslaktır.\nBu seçenekler çalıştırıldığında açıklayıcı hata gösterilir.',wraplength=850).pack(anchor='w',pady=12)
        self.summary = ttk.Label(output,text='Henüz işlem yapılmadı.'); self.summary.pack(anchor='w',pady=8)
        columns=('epok','x','y','z','satellites','code_rms_m')
        self.table=ttk.Treeview(output,columns=columns,show='headings',height=9)
        for col in columns:
            self.table.heading(col,text=col); self.table.column(col,width=125,stretch=True)
        self.table.pack(fill='both',expand=True)
        scrollbar=ttk.Scrollbar(output,orient='vertical',command=self.table.yview); scrollbar.pack(side='right',fill='y')
        self.table.configure(yscrollcommand=scrollbar.set)
        self.log=tk.Text(output,height=6,bg='#172c43',fg='#dcebf7',font=('DejaVu Sans Mono',9),state='disabled')
        self.log.pack(fill='x',pady=10)
        ttk.Label(about,text='Tokat Gaziosmanpaşa Üniversitesi\nDr. Furkan KARLITEPE\nfurkan.karlitepe@gop.edu.tr\n\nGPS Precise Point Positioning — v2.0\n\nDeneysel float PPP: çift frekanslı kod/faz, iyonosferden bağımsız ve Melbourne–Wübbena\nkombinasyonları, uydu saati, Lagrange yörünge enterpolasyonu, Dünya dönüşü düzeltmesi,\nyükseklik ağırlıkları ve sıralı kestirim.\n\nTroposfer, anten ve relativistik düzeltmeler ile hassas saat ürünü uygulanmadığı için\nsonuçlar araştırma amaçlıdır; jeodezik doğruluk garantisi yoktur.\n\nÖzgün kaynaklar legacy/ klasöründe eksiksiz korunmuştur.',justify='left',wraplength=850).pack(anchor='w')
        self.status = ttk.Label(shell,text='Hazır'); self.status.pack(anchor='w',pady=(12,4))
        self.progress=ttk.Progressbar(shell,maximum=100); self.progress.pack(fill='x')
        self.button=ttk.Button(shell,text='İşlemi başlat →',command=self.start); self.button.pack(anchor='e',pady=(10,0))
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.after(100,self.poll)
    def load_example(self):
        root=Path(__file__).resolve().parents[2]
        files=sorted(root.glob('*.SP3'))
        rinex=list(root.glob('*.rnx'))
        if len(files)==3 and rinex:
            for key,path in zip(('sp3_before','sp3_day','sp3_after','rinex'),[*files,rinex[0]]):
                self.fields[key].value.set(str(path))
        else:
            messagebox.showinfo('Örnek veri','Proje kökünde üç SP3 ve bir RINEX dosyası bulunamadı.')
    def add_fcb(self):
        from tkinter.filedialog import askopenfilenames
        for path in askopenfilenames(): self.fcb.insert('end',path)
    def remove_fcb(self):
        for index in reversed(self.fcb.curselection()): self.fcb.delete(index)
    def start(self):
        if self.busy: return
        try:
            values={k:field.value.get().strip() for k,field in self.fields.items()}
            if not values['output']: raise ValueError('Çıktı klasörünü seçin.')
            config=ProcessingConfig(**values,elevation_mask=float(self.mask.get()),technique=self.technique.get(),fcb_files=list(self.fcb.get(0,'end')))
            config.validate()
        except (ValueError,NotImplementedError) as exc:
            messagebox.showerror('Girdileri kontrol edin',str(exc)); return
        self.busy=True; self.button.configure(state='disabled'); self.progress['value']=0
        self.tabs.select(2)
        def worker():
            try:
                result=run(config,lambda p,m:self.events.put(('progress',p,m)))
                self.events.put(('done',result,config.output))
            except Exception as exc:
                self.events.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()
    def poll(self):
        try:
            while True:
                event=self.events.get_nowait()
                if event[0]=='progress':
                    self.progress['value']=event[1]; self.status.configure(text=event[2])
                    self.log.configure(state='normal'); self.log.insert('end',event[2]+'\n'); self.log.see('end'); self.log.configure(state='disabled')
                else:
                    self.busy=False; self.button.configure(state='normal')
                    if event[0]=='error':
                        self.status.configure(text='İşlem başarısız'); messagebox.showerror('İşlem hatası',event[1])
                    else:
                        result=event[1]
                        for item in self.table.get_children(): self.table.delete(item)
                        for _,row in result.positions.iterrows():
                            self.table.insert('', 'end',values=[f'{row[c]:.3f}' for c in self.table['columns']])
                        self.summary.configure(text=f'{len(result.positions)} epok çözüldü · CSV ve Excel: {event[2]}')
        except queue.Empty: pass
        self.after(100,self.poll)
    def close(self):
        if self.busy:
            messagebox.showinfo('İşlem sürüyor','Çıktıların tamamlanması için işlemin bitmesini bekleyin.'); return
        self.destroy()

def main():
    Application().mainloop()
