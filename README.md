# TOGU GPS Processing

GPS Precise Point Positioning (PPP) araştırma projesi. Kod/faz gözlemlerinden deneysel **float PPP** çözümü üretir. Masaüstü arayüzü ve komut satırı aynı işlem servisini kullanır.

## Kurulum ve çalıştırma

Python 3.10 veya üzeri:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m gps_processing --gui
```

Linux'ta masaüstü arayüzü için Tk desteği (`python3-tk`) ve bir grafik oturumu gerekir. Arayüzde **Projedeki örnek verileri yükle** düğmesi depodaki dosyaları seçer. Veri dosyaları, işlem ayarları, sonuçlar ve proje bilgileri ayrı sekmelerdedir. İşlem arka planda yürütülür; ilerleme, hatalar ve epok sonuçları gösterilir.

Komut satırından örnek veri:

```bash
python -m gps_processing \
  --sp3-before IGS0OPSFIN_20233260000_01D_15M_ORB.SP3 \
  --sp3-day IGS0OPSFIN_20233270000_01D_15M_ORB.SP3 \
  --sp3-after IGS0OPSFIN_20233280000_01D_15M_ORB.SP3 \
  --rinex MADR00ESP_R_20233270000_01D_30S_MO.rnx \
  --elevation-mask 10 \
  --output results
```

Eski girişler de kullanılabilir: `python "TOGU-GPS Processing.py"` arayüzü açar; `python karlitepe_processing.py` aynı CLI argümanlarını kabul eder. Modülleri import etmek işlem başlatmaz veya pencere açmaz.

## Modül yapısı

| Bölüm | Dosya | Sorumluluk |
|---|---|---|
| Sabitler | `gps_processing/constants.py` | Işık hızı, frekanslar, elipsoit ve Dünya dönüşü |
| Veri sözleşmeleri | `gps_processing/models.py` | Girdiler, doğrulama, gözlemler ve sonuçlar |
| SP3 | `gps_processing/io/sp3.py` | GPS yörünge/saat okuma; gerçek tarihlerle gün birleştirme |
| RINEX 3 | `gps_processing/io/rinex.py` | Başlığa göre gözlem alanları; L1/L2/L5 fazlarının metreye dönüşümü |
| Kombinasyonlar | `gps_processing/processing/combinations.py` | Ionosphere-free ve Melbourne–Wübbena |
| Yörünge | `gps_processing/processing/orbit.py` | Yayım zamanı, 10 noktalı 9. derece Lagrange, saat enterpolasyonu, Dünya dönüşü |
| Geometri | `gps_processing/processing/geometry.py` | Yükseklik açısı, öncül kovaryans |
| Konum çözümü | `gps_processing/processing/ppp.py` | Sıralı ağırlıklı kod/faz kestirimi; alıcı saati ve uydu belirsizlikleri |
| Dışa aktarım | `gps_processing/io/export.py` | CSV, çok sayfalı Excel ve JSON özet |
| İşlem servisi | `gps_processing/pipeline.py` | Tüm aşamaların açık veri akışı ve ilerleme olayları |
| CLI | `gps_processing/cli.py` | Dosya yolları ve işlem seçenekleri |
| Arayüz | `gps_processing/ui/app.py`, `widgets.py` | Tk masaüstü sekmeleri, dosya seçimi, kuyrukla güvenli arka plan işlemi |

Kütüphane kullanımı:

```python
from pathlib import Path
from gps_processing.models import ProcessingConfig
from gps_processing.pipeline import run

config = ProcessingConfig(
    Path('before.SP3'), Path('day.SP3'), Path('after.SP3'),
    Path('observations.rnx'), output=Path('results'), elevation_mask=10,
)
result = run(config, progress=lambda percent, message: print(percent, message))
print(result.positions)
```

## Korunan bölümler ve değişiklikler

Yedi özgün Python dosyasının tamamı **`legacy/` altında değiştirilmeden saklanmıştır**. Tez, resimler ve örnek veri dosyaları korunmuştur. Kök dizindeki eski modül adları yeni modüllere uyumluluk girişleridir. Eski API'de RINEX okuyucusu yaklaşık koordinatı DataFrame metadata'sında taşır; PPP çağrısında `approximate_xyz=` açıkça da verilebilir. Ara Excel dosyalarına bağımlılık kaldırılmıştır; ara çıktılar artık yalnız dışa aktarım aşamasında üretilir.

Sabit 32 uydu/900 saniye varsayımıyla SP3 satır sayma yerine gerçek epoklar okunur. Saat tahminindeki Random Forest yerine deterministik doğrusal enterpolasyon kullanılır; Lagrange hesabı büyük mutlak zamanlarla polinom üretmek yerine merkezlenmiş barycentric yöntemle yapılır. Bu nedenle sonuçlar eski betiklerle bit düzeyinde aynı değildir. Konum hesabında uydu/epok eşleştirmesi anahtarlarla yapılır, düşük yükseklikli gözlemler gerçekten elenir, tüm çözülen epoklar saklanır. Kestirim saat durumunu her epokta serbest bırakır; faz kilit kaybı işareti, 120 saniyeden uzun boşluk veya 5 metreden büyük MW değişiminde ilgili belirsizliği sıfırlar. Bu eşikler temel araştırma varsayımlarıdır.

İyonosfer/troposfer dosya alanları, PPP-AR/SDBS, Machine Learning ve FCB referans dosyaları arayüzde korunmuştur. **Özgün kodda bunların çözücüleri yoktu**; yeni arayüz bunları tamamlanmış gibi çalıştırmaz ve seçildiklerinde açıklayıcı hata verir. RINEX 2 de desteklenmez. Harici model seçenekleri, ilgili model eklendiğinde `ProcessingConfig` üzerinden işlem servisine bağlanabilir.

## Çıktılar

Seçilen klasörde `results.xlsx`, `summary.json` ve her tablo için CSV bulunur:

- `combine_sp3`, `obs3_gps`, `mw_ion_gps`, `xyz0_apprx`
- `result_gps`, `sat3_df`, `cll` (öncül), `cll_result` (son kovaryans)
- `dx_1` (son epok durum güncellemesi), `n3_amb` (metre cinsinden float belirsizlik)
- `xyz_result` (tüm epok koordinatları, alıcı saati, uydu sayısı, kod RMS)
- `diagnostics` (çözülen/atlanmış epoklar)

SP3 koordinatları girişte km, çözümde metredir. SP3 saatleri mikrosaniyedir; alıcı saat çıktısı saniyedir. `epok` ölçüm günü gece yarısından itibaren saniyedir. Eksik gözlemler sıfır yerine eksik değer olarak tutulur. Mevcut örnek RINEX dosyası yaklaşık 12 dakikalık gözlem içerir; başlıktaki tüm güne ait kapsam gerçek kayıt sayısı değildir.

## Doğrulama ve bilimsel sınırlar

```bash
python -m unittest discover -s tests -v
```

Testler kombinasyonların iyonosferi gidermesini, polinom enterpolasyonunu, yükseklik geometrisini, bilinen sentetik konum/saat çözümünü, değişken uydu sayılı SP3 okumayı, gerçek örnek dosyaları ve desteklenmeyen modları kapsar.

Bu çözüm üretim seviyesinde tam PPP değildir. Troposfer, anten faz merkezi, phase wind-up, relativistik etkiler, gelgit, kod bias düzeltmeleri ve ayrı hassas saat ürünü uygulanmamıştır. SP3 saatleri ile kod RMS/koordinat değerleri jeodezik doğruluk kanıtı değildir. Tam PPP-AR için FCB/bias modelleri ve tamsayı belirsizlik çözümü ayrıca gerekir.

Tokat Gaziosmanpaşa Üniversitesi · Dr. Furkan KARLITEPE · furkan.karlitepe@gop.edu.tr
