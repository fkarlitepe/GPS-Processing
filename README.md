# TOGU GNSS Processing

GNSS (GPS + Galileo) Precise Point Positioning (PPP / PPP-AR) araştırma projesi.

**v2.1:** Kestirim noktalarına iterasyonlu **genişletilmiş Kalman filtresi** (statik/kinematik, ZWD kestirimi, IGG-III robust ve adaptif güncelleme), **LAMBDA + kısmi AR** ve iki **makine öğrenmesi** bileşeni (artıklardan öğrenilen gözlem varyansı, Monte Carlo ile eğitilmiş AR doğrulayıcı) eklendi; arayüze yöntem ayarları, grafikler ve **yöntem karşılaştırması** sekmesi geldi. Denklemler, örnek sonuçlar ve deney tasarımı: [EKF ve ML belgesi](docs/EKF_ML.md).

**v2.5:** Integer phase clock (IPC, Laurichesse) kullanıcı modeli eklendi (`--technique PPP-AR/IPC`). SDBS/DCM, Shi & Gao (2014) karşılaştırmasına göre gözden geçirildi; referans uydu değişiminde belirsizlikler artık sıfırlanmıyor, tamsayı kaydırmayla korunuyor. Ayrıntılar: [PPP-AR belgesi](docs/PPP_AR.md).

**v2.8 (GNSS):** GPS (L1/L2) ve Galileo (E1/E5a) seçimli çözüm (`--systems G E`, arayüzde onay kutuları). RINEX 2/3, SP3, CLK, ANTEX, OSB ve FCB okuyucuları Galileo'yu da okur. Her sistemin kendi alıcı saati/biasları ve kendi referans uydusu vardır; FCB kestirimi sistem başına en az 5 istasyon ister. Ayrıntılar: [GNSS belgesi](docs/GNSS.md).

**v2.7:** RINEX 2.11 okuma; katı yer gelgiti (IERS 2010); OSB/FCB ürünlerinde eksik uydu saati düzeltildi (v2.6 hatası). `Data/network` istasyonları arayüzde otomatik listelenir; PPP-AR'de ürün yüklenmezse FCB'ler bu istasyonlardan kestirilip kullanılır. Referans ağdan (en az 5 istasyon) FCB kestirimi: Li & Zhang (2012) toplu çözüm, EKF ve EKF + ML kestiricileri, istasyon dışarıda bırakmalı doğrulama, GENFCB çıktısı (`--fcb-stations`). Ayrıntılar: [FCB belgesi](docs/FCB.md).

**v2.6:** Gerçek ürünler: RINEX CLK saatleri (`--clock-file`), Bias-SINEX kod+faz OSB (`--ar-products *.BIA`) ve GENFCB FCB (`*.fcb`) AR ürünleri, ANTEX PCO/PCV (`--antex`) ve faz wind-up; tek günlük SP3 desteği. Ayrıntılar: [PPP-AR belgesi](docs/PPP_AR.md).

Ayrıca (Banville et al. 2008): Bias-SINEX DSB/OSB dosyasıyla C1C/C2L gibi kodlar C1W/C2W'ye hizalanır (`--code-biases`), alıcı milisaniye saat sıçramaları algılanıp fazdan giderilir (varsayılan açık, `--no-clock-jump-repair`).

**v2.4:** LAMBDA'ya alternatif tamsayı kestiricileri eklendi: dekorelasyonlu integer bootstrapping (+PAR) ve TCAR/CIR ilkesine göre kademeli yuvarlama (`--integer-method bootstrap|cascade`). Başarı oranları Ps-LAMBDA formülleriyle raporlanıyor; AR için ML doğrulayıcı scikit-learn ile seçildi. Ayrıntılar: [EKF ve ML belgesi](docs/EKF_ML.md).

**v2.2:** IONEX küresel iyonosfer haritası (tek tabaka modeli, IF kombinasyonunda ikinci derece terim) ve SINEX_TRO istasyon ZTD ürünü (a-priori ZTD + Chen & Herring gradyanları) hesaba katılır (`--ionosphere`, `--troposphere`). Arayüz başlıkları ve açıklamaları İngilizcedir. Ayrıntılar: [Atmosphere products](docs/ATMOSPHERE.md).

PPP-AR/SDBS ve PPP-AR/DCM için ayrı deneysel kullanıcı modelleri vardır. Ürün sözleşmesi, literatür karşılaştırması, denklemler ve bilimsel sınırlar için [PPP-AR teknik belgesi](docs/PPP_AR.md). AR için normalize saat/bias JSON ürünü gerekir; depoda bu ürün yoktur.
 Kod/faz gözlemlerinden deneysel **float PPP** çözümü üretir. Masaüstü arayüzü ve komut satırı aynı işlem servisini kullanır.

## Kurulum ve çalıştırma

Python 3.10 veya üzeri:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m gps_processing --gui
```

Linux'ta masaüstü arayüzü için Tk desteği (`python3-tk`) ve bir grafik oturumu gerekir. Arayüzde **Data klasöründeki verileri seç** düğmesi Data içindeki dosyaları seçer. Veri dosyaları, işlem ayarları, sonuçlar ve proje bilgileri ayrı sekmelerdedir. İşlem arka planda yürütülür; ilerleme, hatalar ve epok sonuçları gösterilir.

Komut satırından örnek veri:

```bash
python -m gps_processing \
  --sp3-before IGS0OPSFIN_20233260000_01D_15M_ORB.SP3 \
  --sp3-day IGS0OPSFIN_20233270000_01D_15M_ORB.SP3 \
  --sp3-after IGS0OPSFIN_20233280000_01D_15M_ORB.SP3 \
  --rinex madr3270.23o \
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
| FCB kestirimi | `gps_processing/processing/fcb/` | Ağdan WL/NL UFO kestirimi (batch, EKF, EKF+ML), doğrulama, GENFCB yazımı |
| Saat dosyası | `gps_processing/io/clock_rinex.py` | RINEX CLK 30 s / 5 s uydu saatleri |
| AR ürünleri | `gps_processing/io/osb_products.py`, `io/fcb_products.py` | Bias-SINEX kod+faz OSB ve GENFCB FCB → IF kod/faz ve MW düzeltmeleri |
| Anten modelleri | `gps_processing/io/antex.py`, `processing/antenna.py` | ANTEX PCO/PCV, nominal yaw, faz wind-up |
| Kod biasları | `gps_processing/io/bias_sinex.py` | Bias-SINEX DSB/OSB okuma; C1C/C2L… kodlarını C1W/C2W'ye hizalama |
| Saat sıçraması | `gps_processing/processing/clock_jump.py` | 1 ms alıcı saat sıçramalarının algılanması ve fazdan giderilmesi |
| Kombinasyonlar | `gps_processing/processing/combinations.py` | Ionosphere-free ve Melbourne–Wübbena |
| Yörünge | `gps_processing/processing/orbit.py` | Yayım zamanı, 10 noktalı 9. derece Lagrange, saat enterpolasyonu, Dünya dönüşü |
| Geometri | `gps_processing/processing/geometry.py` | Yükseklik açısı, öncül kovaryans |
| Konum çözümü (v2.0 temel) | `gps_processing/processing/ppp.py` | Sıralı ağırlıklı kod/faz kestirimi; alıcı saati ve uydu belirsizlikleri |
| EKF çekirdeği | `gps_processing/processing/estimation/` | İterasyonlu EKF, süreç gürültüsü, IGG-III robust, adaptif fading |
| EKF float PPP | `gps_processing/processing/ppp_ekf.py` | XYZ, saat, ZWD, IF belirsizlikleri; artık tablosu |
| Fiziksel modeller | `processing/troposphere.py`, `processing/cycle_slip.py` | Saastamoinen + Niell (varsayılan) veya Chao eşleme; LLI/boşluk/MW/GF kayma tespiti |
| PPP-AR | `gps_processing/processing/ar/` | SDBS/DCM filtresi, ILS, LAMBDA dekorelasyonu, bootstrapping PAR |
| Makine öğrenmesi | `gps_processing/processing/ml/` | GBRT stokastik model, lojistik AR doğrulayıcı (numpy) |
| Analiz | `gps_processing/analysis.py` | ENU hataları, yakınsama süresi, yöntem istatistikleri |
| Dışa aktarım | `gps_processing/io/export.py` | CSV, çok sayfalı Excel ve JSON özet |
| İşlem servisi | `gps_processing/pipeline.py` | Tüm aşamaların açık veri akışı ve ilerleme olayları |
| CLI | `gps_processing/cli.py` | Dosya yolları ve işlem seçenekleri |
| Arayüz | `gps_processing/ui/app.py`, `widgets.py` | Tk masaüstü sekmeleri, dosya seçimi, kuyrukla güvenli arka plan işlemi |

EKF seçenekleri ve yöntem karşılaştırması (CLI):

```bash
python -m gps_processing --sp3-before Data/…260….SP3 --sp3-day Data/…270….SP3 --sp3-after Data/…280….SP3 \
  --rinex Data/madr3270.23o --estimator ekf --ml-weighting \
  --reference-xyz 4849202.3940 -360328.9929 4114913.1862 \
  --compare 'Sequential LS (v2.0)' 'EKF' 'Adaptive + robust EKF' 'EKF + ML weighting' --output results
```

Seçenekler: `--estimator sequential|ekf|adaptive-ekf`, `--dynamics static|kinematic`, `--no-troposphere`, `--no-physical-corrections`, `--no-robust`, `--integer-method ils|lambda`, `--ar-success-rate`, `--ar-validation ratio|ml|ratio+ml`, `--ml-weighting`, `--ml-validator Data/models/ar_validator.json`. `sequential` v2.0 sonucunu değiştirmeden üretir.

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

FCB referans dosya alanı korunmuştur; FCB kestirimi henüz uygulanmadı. PPP-AR/SDBS ve PPP-AR/DCM araştırma çözücüleri ayrı modüllerdedir. Makine öğrenmesi ayrı bir yöntem değil, EKF ve AR kestirimine eklenen bileşenler olarak uygulanmıştır (docs/EKF_ML.md); özgün Random Forest kaynağı legacy altında korunur. RINEX 2 de desteklenmez. Harici model seçenekleri, ilgili model eklendiğinde `ProcessingConfig` üzerinden işlem servisine bağlanabilir.

## Çıktılar

Her işlem, seçilen çıktı klasörünün (varsayılan `results/`) altında kendi klasörüne yazılır: `<yıl>_<yılın günü>_<SSDDss>` (işlemin başlama zamanı, örneğin `results/2026_302_143512/`; aynı saniyede ikinci işlem `_2` eki alır). Karşılaştırmalar bu klasörün `comparison/`, FCB kestirimi `fcb/` alt klasörüne yazılır; işlemler birbirine karışmaz. Klasörde `results.xlsx`, `summary.json` ve her tablo için CSV bulunur:

- `combine_sp3`, `obs3_gps`, `mw_ion_gps`, `xyz0_apprx`
- `result_gps`, `sat3_df`, `cll` (öncül), `cll_result` (son kovaryans)
- `dx_1` (son epok durum güncellemesi), `n3_amb` (metre cinsinden float belirsizlik)
- `xyz_result` (tüm epok koordinatları, alıcı saati, uydu sayısı, kod RMS)
- `diagnostics` (çözülen/atlanmış epoklar; EKF'de NIS, adaptif katsayı, ağırlığı düşürülen uydular)
- `residuals` (EKF: epok/uydu bazında kod/faz artıkları, yükseklik, SNR, yay uzunluğu — ML eğitim verisi)
- `statistics` (ENU RMS, son hata, yakınsama, fix oranı); `ml_models.json` (öğrenilen modeller)
- Karşılaştırmada `comparison/<yöntem>/` klasörleri ve `comparison/statistics.csv`

SP3 koordinatları girişte km, çözümde metredir. SP3 saatleri mikrosaniyedir; alıcı saat çıktısı saniyedir. `epok` ölçüm günü gece yarısından itibaren saniyedir. Eksik gözlemler sıfır yerine eksik değer olarak tutulur. Mevcut örnek RINEX dosyası yaklaşık 12 dakikalık gözlem içerir; başlıktaki tüm güne ait kapsam gerçek kayıt sayısı değildir.

## Doğrulama ve bilimsel sınırlar

```bash
python -m unittest discover -s tests -v
```

Testler kombinasyonların iyonosferi gidermesini, polinom enterpolasyonunu, yükseklik geometrisini, bilinen sentetik konum/saat çözümünü, değişken uydu sayılı SP3 okumayı, gerçek örnek dosyaları ve desteklenmeyen modları kapsar. `tests/test_estimation.py` EKF'nin doğrusal LS ile eşdeğerliğini, ZWD/saat/konum kestirimini, robust aykırı gözlem reddini, kinematik izlemeyi, LAMBDA'nın kaba kuvvet ILS ile eşleşmesini, bootstrapping alt sınırını, PAR'ı, ML öğrenicilerini ve örnek veri karşılaştırmasını test eder.

Bu çözüm üretim seviyesinde tam PPP değildir. EKF kestiricileri Saastamoinen troposferi, ZWD kestirimini ve periyodik relativistik terimi uygular; anten faz merkezi, phase wind-up, gelgit, kod bias düzeltmeleri ve ayrı hassas saat ürünü hâlâ uygulanmamıştır. SP3 saatleri ile kod RMS/koordinat değerleri jeodezik doğruluk kanıtı değildir. PPP-AR çekirdeği eklendi; gerçek ürün okuyucuları ve tam fiziksel PPP düzeltmeleri hâlâ gereklidir. Ayrıntılar docs/PPP_AR.md içindedir.

Tokat Gaziosmanpaşa Üniversitesi · Dr. Furkan KARLITEPE · furkan.karlitepe@gop.edu.tr


## Data klasörü ile yeni işlem

Her yeni işlemden önce gerekli girdi dosyalarını proje kökündeki `Data/` klasörüne koyun. PPP için önceki/ölçüm/sonraki gün SP3 ve RINEX 3 gözlem dosyası; PPP-AR için ayrıca yönteme uygun JSON saat/bias ürünü gerekir. Alt klasörler kullanılabilir. Arayüz dosya pencereleri Data içinde açılır; arayüz ve CLI Data dışındaki girdileri reddeder. Çıktı klasörü serbesttir.

“Data klasöründeki verileri seç” önce eski seçimleri temizler, tam üç SP3 ve bir gözlem varsa SP3 başlık tarihlerine göre seçer. Birden çok veri setinde dosyaları Seç düğmeleriyle belirleyin. AR JSON dosyasını İşlem ayarları sekmesinde seçin. Veriler otomatik silinmez veya taşınmaz. CLI örneklerinde dosya yollarının başına `Data/` ekleyin; mevcut örnekleri kullanmak için önce bu dosyaları Data içine kopyalayın.
