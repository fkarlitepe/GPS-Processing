# TOGU GNSS Processing

TOGU GNSS Processing, GPS, Galileo ve BeiDou gözlemleriyle hassas nokta konumlama (PPP) ve tamsayı belirsizlik çözümlü PPP (PPP-AR) yapan bir araştırma yazılımıdır. Tokat Gaziosmanpaşa Üniversitesi'nde geliştirilmiştir. Float PPP ve üç PPP-AR kullanıcı modeli aynı genişletilmiş Kalman filtresini (EKF) kullanır. Makine öğrenmesi ayrı bir yöntem değildir; EKF'nin ve tamsayı kararının belirli adımlarını destekleyen bileşenler olarak eklenmiştir. Masaüstü arayüzü ve komut satırı aynı işlem servisini çalıştırır.

## Paper and reproducibility

This repository contains the software and scripts of the paper *Supporting GPS+Galileo PPP Ambiguity Resolution with an Extended Kalman Filter and Machine Learning* (Karlıtepe, submitted to the ASCE Journal of Surveying Engineering). It is released under the MIT License; see `CITATION.cff` for citation.

The experiments use IGS station MADR on November 23, 2023 (day 327). The GNSS observations and the MGEX products (COD, GRG, WUM, GFZ) are not redistributed here. Download them from the IGS data centres into the paths listed in `experiments/common.py`: the MADR RINEX file into `Data/obs/`, the network RINEX files (`experiments/network_stations.csv`) into `Data/network/`, the products into `Data/products/2023327/<centre>/`, the IGS SINEX file into `Data/sinex/`, and the ANTEX, BLQ and ERP files into `Data/models/`. Then run:

```bash
python3 -m experiments.nl_test                      # narrow-lane product test
python3 -m experiments.run_grid products            # Experiment A: analysis-centre products
python3 -m experiments.run_network_fcb              # Experiment B, step 1: FCBs from the station network
python3 -m experiments.run_grid network results/experiments/network_fcb/fcb   # Experiment B, step 2
python3 -m experiments.run_grid ablation            # contribution of each component
python3 experiments/analyse.py products|network|ablation|nl   # tables, statistics and figures
python3 experiments/station_map.py                  # station map (requires PyGMT)
```

Set `OMP_NUM_THREADS=1` when the grids run in parallel. Results are written to `results/experiments/`.

Uydu seçimi ve kalite kontrolü deneyleri bu makaleden ayrıdır: `python3 -m experiments.run_selection` (DOP ve ağırlıklı DOP, sekiz istasyon), `python3 -m experiments.run_selection_rev2 run GE|GEC` (üç gün, GPS+Galileo ve GPS+Galileo+BeiDou) ve `python3 -m experiments.run_quality` (aykırı gözlem testleri ve robust kestirim). Analizleri `experiments/analyse_selection.py` ve `experiments/analyse_selection_rev2.py` üretir.

## Temel özellikler

- **Gözlem ve ürünler:** RINEX 2.11 ve RINEX 3 gözlemleri; SP3 yörüngeleri, RINEX CLK saatleri, ANTEX, Bias-SINEX (kod ve faz OSB), GENFCB, ORBEX, IONEX, SINEX_TRO, BLQ ve ERP dosyaları.
- **Fiziksel modeller:** Saastamoinen + Niell troposferi ve ZWD kestirimi; katı yer, okyanus yüklemesi ve kutup gelgitleri; uydu ve alıcı anten PCO/PCV; faz wind-up; Shapiro gecikmesi; alıcı saat sıçramalarının giderilmesi.
- **Kestirim:** Float PPP ve PPP-AR için tek EKF çekirdeği; statik veya kinematik çözüm; IGG-III, Huber ve Danish robust kestirimi; global χ², Baarda ve Pope testleri; gözlem grubu başına uyarlamalı varyans faktörleri.
- **PPP-AR:** SDBS, DCM ve IPC kullanıcı modelleri. Tamsayı kararı LAMBDA, bootstrapping veya kademeli yuvarlama ile verilir. Karar zinciri Dong–Bock sıralaması, başarı oranına göre kısmi çözüm, sabit hata oranlı oran testi (FFRT) ve uydu dışlamadan oluşur.
- **Makine öğrenmesi:** Artıklardan öğrenilen gözlem varyansı, sapmalı float belirsizliklere duyarlı kabul testi ve kendi kendini eğiten çözülebilirlik modeli.
- **Ürün denetimi ve FCB:** Dar şerit (NL) teşhis testi; bias, saat ve yörünge dosyalarının aynı merkezden geldiğinin denetimi; referans ağdan toplu, EKF ve EKF+ML yöntemleriyle FCB kestirimi ve istasyon dışarıda bırakmalı doğrulama.
- **Uydu seçimi:** Float PPP'de her epokta k uydunun DOP veya ağırlıklı DOP ölçütüne göre seçimi (FIBO-GNSS, geri eleme, yerel arama ve diğer yöntemler).

## Kurulum

Python 3.10 veya üzeri gerekir. Linux'ta arayüz için ayrıca Tk paketi (`python3-tk`) kurulmalıdır.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Zorunlu bağımlılıklar numpy, pandas, openpyxl ve scipy'dir. `pip install .[ml]` komutu scikit-learn ve joblib'i, `pip install .[config]` komutu YAML ve TOML desteğini ekler. scikit-learn kurulu değilse yazılım bağımlılıksız bir lojistik doğrulayıcı kullanır.

## Çalıştırma

**Arayüz.** Proje klasöründe aşağıdaki komut çalıştırılır:

```bash
python3 "TOGU-GPS Processing.py"
```

Aynı arayüz `python3 -m gps_processing --gui` komutuyla da açılır. Veri dosyaları, işlem ayarları, sonuçlar ve proje bilgileri ayrı sekmelerdedir. İşlem arka planda yürür; ilerleme ve hatalar ekranda gösterilir.

**Komut satırı.** Depodaki örnek veriyle float PPP çözümü:

```bash
python3 -m gps_processing \
  --sp3-before Data/IGS0OPSFIN_20233260000_01D_15M_ORB.SP3 \
  --sp3-day Data/IGS0OPSFIN_20233270000_01D_15M_ORB.SP3 \
  --sp3-after Data/IGS0OPSFIN_20233280000_01D_15M_ORB.SP3 \
  --rinex Data/madr3270.23o --elevation-mask 10 --output results
```

Başlıca seçenekler şunlardır: `--technique PPP|PPP-AR/SDBS|PPP-AR/DCM|PPP-AR/IPC`, `--systems G E C`, `--ar-products`, `--clock-file`, `--antex`, `--estimator sequential|ekf|adaptive-ekf`, `--dynamics static|kinematic`, `--ar-validation ffrt|ratio|ml|ffrt+ml|ratio+ml`, `--selection`, `--diagnose-nl`. Birden fazla yöntem `--compare` ile aynı veride karşılaştırılabilir. Tam liste için `python3 -m gps_processing --help` kullanılır.

**Ayar dosyası.** `--write-config run.json` bütün ayarları bir dosyaya yazar, `--config run.json` bu dosyayı okur. Komut satırında verilen seçenekler dosyadaki değerlerin yerine geçer.

**Python'dan kullanım.**

```python
from pathlib import Path
from gps_processing.models import ProcessingConfig
from gps_processing.pipeline import run

config = ProcessingConfig(
    Path('Data/before.SP3'), Path('Data/day.SP3'), Path('Data/after.SP3'),
    Path('Data/observations.rnx'), output=Path('results'), elevation_mask=10,
)
result = run(config, progress=lambda percent, message: print(percent, message))
print(result.positions)
```

## Veri klasörü

Girdi dosyaları proje kökündeki `Data/` klasöründe bulunmalıdır; arayüz ve komut satırı bu klasörün dışındaki girdileri kabul etmez. Alt klasörler kullanılabilir. PPP için ölçüm gününün SP3 dosyası ve gözlem dosyası gerekir; önceki ve sonraki günün SP3 dosyaları enterpolasyonu gün sınırlarında iyileştirir. PPP-AR için ayrıca bias veya FCB ürünü gerekir. Arayüzdeki **Data klasöründeki verileri seç** düğmesi uygun dosyaları SP3 başlık tarihlerine göre seçer. Yazılım bu klasördeki dosyaları silmez veya taşımaz.

Depoda yalnızca örnek veri bulunur: MADR istasyonunun 23 Kasım 2023 tarihli, 30 saniye aralıklı RINEX 2.11 dosyası ve üç günlük IGS SP3 dosyaları. Deney verileri ve ürünler yerelde tutulur, git'e eklenmez.

## Çıktılar

Her işlem, çıktı klasörünün (varsayılan `results/`) altında başlama zamanıyla adlandırılan ayrı bir klasöre yazılır; örneğin `results/2026_302_143512/`. Klasörde çok sayfalı `results.xlsx`, `summary.json` ve her tablo için bir CSV dosyası bulunur. Başlıca tablolar şunlardır:

- `xyz_result`: epok koordinatları, alıcı saati ve uydu sayısı;
- `residuals`: epok ve uydu bazında kod ve faz artıkları;
- `diagnostics`: çözülen ve atlanan epoklar, NIS ve robust ağırlıklar;
- `statistics`: ENU RMS hataları, yakınsama süresi ve fix oranı.

Yöntem karşılaştırmaları `comparison/`, FCB kestirimi `fcb/` alt klasörüne yazılır. `epok` sütunu, ölçüm gününün başından itibaren geçen saniyedir.

## Klasör yapısı

| Klasör veya dosya | İçerik |
|---|---|
| `TOGU-GPS Processing.py` | Arayüzü açan başlatıcı |
| `gps_processing/` | Yazılımın tamamı: dosya okuyucuları (`io/`), işlem çekirdeği (`processing/`), arayüz (`ui/`), komut satırı (`cli.py`) ve işlem servisi (`pipeline.py`) |
| `experiments/` | Makale deneylerinin betikleri |
| `tests/` | Birim testleri |
| `docs/` | Yöntem belgeleri |
| `manuscript/` | Makale kaynakları ve derleme betiği; çıktı belgeler git'e eklenmez |
| `Data/` | Örnek veri ve yerel deney verileri |
| `archive/v1_2023/` | 2023'teki ilk sürümün özgün betikleri; değiştirilmeden saklanır, yazılım bunları kullanmaz |

`processing/` klasöründe EKF çekirdeği `engine.py` ve `estimation/`, PPP-AR `ar/`, FCB kestirimi `fcb/`, makine öğrenmesi `ml/`, uydu seçimi `selection.py`, NL teşhisi `diagnostics.py` dosyalarındadır. v2.0'ın sıralı en küçük kareler çözücüsü karşılaştırma için `sequential.py` içinde değiştirilmeden tutulur.

## Yöntem belgeleri

| Belge | Konu |
|---|---|
| [PPP_AR.md](docs/PPP_AR.md) | PPP-AR kullanıcı modelleri, ürün sözleşmesi ve denklemler |
| [EKF_ML.md](docs/EKF_ML.md) | EKF, robust kestirim, kalite testleri ve makine öğrenmesi bileşenleri |
| [FCB.md](docs/FCB.md) | Referans ağdan FCB kestirimi ve doğrulama |
| [GNSS.md](docs/GNSS.md) | GPS, Galileo ve BeiDou işleme |
| [MODELS.md](docs/MODELS.md) | Fiziksel modeller ve NL teşhisi |
| [ATMOSPHERE.md](docs/ATMOSPHERE.md) | İyonosfer ve troposfer ürünleri |
| [SELECTION.md](docs/SELECTION.md) | Uydu seçimi, DOP ve ağırlıklı DOP analizi |

## Testler

```bash
python3 -m unittest discover tests
```

Testler kombinasyonları, enterpolasyonu, dosya okuyucularını, EKF ile en küçük kareler eşdeğerliğini, robust kestirimi, LAMBDA ve kısmi çözümü, ML bileşenlerini ve örnek veri çözümlerini kapsar.

## Sürüm geçmişi

| Sürüm | Başlıca değişiklik |
|---|---|
| v3.2 | İstatistiksel kalite kontrolü: global χ², Baarda ve Pope testleri; Huber ve Danish robust kestirimi |
| v3.1 | BeiDou desteği; FIBO-GNSS2 ve ileri seçim yöntemleri |
| v3.0 | Float PPP'de uydu seçimi ve DOP / ağırlıklı DOP analizi |
| v2.9 | Literatüre dayalı gözden geçirme: yeni fiziksel modeller, AR karar zinciri, NL teşhisi, tek EKF çekirdeği, büyük FCB ağları |
| v2.8 | GPS ve Galileo seçimli çözüm |
| v2.7 | RINEX 2.11, katı yer gelgiti, referans ağdan FCB kestirimi |
| v2.6 | Gerçek ürünler: RINEX CLK, Bias-SINEX OSB, GENFCB, ANTEX ve faz wind-up |
| v2.5 | IPC kullanıcı modeli |
| v2.4 | Bootstrapping ve kademeli yuvarlama tamsayı kestiricileri |
| v2.2 | IONEX ve SINEX_TRO atmosfer ürünleri |
| v2.1 | EKF, LAMBDA ile kısmi çözüm ve ilk makine öğrenmesi bileşenleri |
| v2.0 | Modüler yapı; ilk sürümün betikleri `archive/v1_2023/` altına alındı |

## Kapsam ve sınırlar

Yazılım bir araştırma aracıdır; sonuçlar ticari veya operasyonel PPP yazılımlarıyla doğrulanmadan jeodezik ürün olarak kullanılmamalıdır. Çözümler son işlem (post-processing) içindir; gerçek zamanlı işleme desteklenmez. PPP-AR'nin başarısı, kullanılan bias ürünlerinin yörünge ve saat ürünleriyle tutarlı olmasına bağlıdır. Bu nedenle yeni bir ürün setiyle çalışmadan önce NL teşhis testinin (`--diagnose-nl`) uygulanması önerilir.

## Lisans ve atıf

MIT lisansı ile dağıtılır (`LICENSE`). Atıf bilgisi `CITATION.cff` dosyasındadır.

Dr. Furkan KARLITEPE · Tokat Gaziosmanpaşa Üniversitesi, Harita Mühendisliği Bölümü
