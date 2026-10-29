# FCB (UFO / UPD) kestirimi — referans ağ

Li & Zhang (2012) *Improving the Estimation of Uncalibrated Fractional Phase Offsets for PPP Ambiguity Resolution* yöntemine göre, en az **5 istasyonun** aynı güne ait RINEX dosyasından uydu FCB'leri hesaplanır. Çıktı, PPP-AR'de doğrudan ürün olarak kullanılabilen GENFCB dosyasıdır (`--ar-products fcb_ekf.fcb`).

## Akış (`gps_processing/processing/fcb/network.py`)

1. **Float PPP:** Her istasyon, kullanıcının da kullanacağı SP3/CLK/ANTEX ve modellerle EKF float PPP ile çözülür. Her epokta IF belirsizliği (`N3`, m), σ'sı, arc numarası ve yükseklik açısı saklanır.
2. **Geniş şerit (WL):** MW kombinasyonunun arc ortalaması (çevrim, σ = std/√n, en az 20 epok) → WL UFO'ları (varsayılan günde bir blok) → kesri `gate` (0.25 çevrim) altındaki arc'ların tamsayı WL'si.
3. **Dar şerit (NL):** Her bloğun (varsayılan 15 dk) sonundaki float IF belirsizliğinden `B_NL = (B_IF − β·N_WL)/λ_NL`; σ_NL < 0.2 çevrim olanlar kullanılır → NL UFO'ları.
4. **Dosya:** `b1 = s_NL − (β/λ_NL)·s_WL`, `b2 = b1 − s_WL`. `io/fcb_products.py` bunları yine `−λ_W·s_WL` (MW) ve `−λ_NL·s_NL` (IF faz) düzeltmelerine çevirir. İşaretler testte tam zincirle doğrulanır: dosya yazılır, okunur, kullanıcının uydular arası WL ve NL belirsizlikleri tamsayı çıkar.

Gözlem denklemi (Li & Zhang denk. 10–12): `b_ij = n_ij + r_i + s_j`. Bir istasyonun UFO'su datumdur (r = 0); tamsayılar 1 çevrimlik belirsizliği taşır.

## Kestiriciler (`gps_processing/processing/fcb/core.py`)

| Yöntem | Mantık |
|---|---|
| `batch` | Makaledeki yöntem: her blokta datum istasyonundan başlayan dairesel ortalamalarla başlangıç, kesri `gate` altındaki belirsizlikleri sabitleme, r ve s için ağırlıklı en küçük kareler, kabul kümesi değişmeyene kadar tekrar (sıralı sabitleme). Bloklar birbirinden bağımsızdır. |
| `ekf` | Aynı denklem, durumu tüm r_i ve s_j olan doğrusal Kalman filtresinde. UFO'lar rastgele yürüyüş (uydu 0.01, alıcı 0.02 çevrim/blok). Tamsayılar tahmin edilen duruma göre sabitlenir (yenilik < 0.25 çevrim ve tahmin σ < 0.2), blok içinde dört tura kadar sıralı. UFO'lar bloklar ve datum boyunca süreklidir; az gözlemli bloklar önceki bloklardan bilgi alır. İlk kez görülen r/s, bilinenlerden dairesel ortalamayla başlatılır. |
| `ekf+ml` | İki EKF geçişi. Arada, gradient boosting (istasyona göre çapraz uydurma, 3 kat) her belirsizliğin artık varyansını kendi float σ'sı, yükseklik açısı ve arc uzunluğundan öğrenir. İkinci geçiş öğrenilen σ'ları ağırlık olarak kullanır; öğrenilen σ ≥ 0.2 çevrim olanları dışarıda bırakır. 40'tan az belirsizlikte EKF sonucu döner. |

**Doğrulama (her yöntem için):** Her istasyon sırayla dışarıda bırakılır, kalanlardan UFO kestirilir, dışarıdaki istasyonun NL belirsizlikleri düzeltilir. Alıcı kısmı, uydular arası farkın yaptığı gibi her blokta dairesel ortalamayla çıkarılır (en az 3 uydu). Kalan kesirlerin RMS'i ve 0.15/0.25 çevrim altındaki oranı `validation_<yöntem>.csv` ve `summary.json` dosyalarına yazılır. `best_method`, en düşük ortalama RMS'e sahip yöntemdir.

**Sentetik ağ sonuçları** (8 istasyon, 14 uydu, 48 blok; düşük açıda büyüyen saçılma ve %6 biaslı belirsizlik; `tests/fcb_simulation.py`):

| Tohum | batch: uydu hatası / LOO RMS | ekf | ekf+ml |
|---|---|---|---|
| 0 | 0.074 / 0.093 | 0.038 / 0.063 | 0.033 / 0.064 |
| 1 | 0.076 / 0.075 | 0.080 / 0.058 | 0.077 / 0.064 |
| 2 | 0.097 / 0.091 | 0.033 / 0.071 | 0.025 / 0.070 |

EKF, bağımsız bloklara göre belirgin biçimde iyi. ML'in katkısı küçük ve her durumda olumlu değil; bu yüzden varsayılan `all`, üçünü birlikte çalıştırıp çapraz doğrulamayla seçer.

## Data/network ve PPP-AR akışı

- `Data/network/` altındaki gözlem dosyaları (RINEX 2.11 veya 3) arayüzde 7. bölüme (FCB estimation) otomatik eklenir; kullanıcı istasyonu `Data/` altında ayrıca durur.
- **PPP** tekniğinde 7. bölüm gizlidir; FCB yalnız PPP-AR için anlamlıdır.
- **PPP-AR:** AR ürün alanına bir FCB (`*.fcb`), OSB (`*.BIA`) ya da JSON ürünü yüklenmişse doğrudan o kullanılır. Alan boşsa ve en az 5 referans istasyon varsa önce FCB'ler bu istasyonlardan kestirilir (`fcb_method`, varsayılan `all`), istasyon dışarıda bırakmalı doğrulamada en iyi yöntemin dosyası seçilir ve PPP-AR onunla çözülür. Kestirim özeti sonuç dosyasında `ar_products_and_settings.fcb_estimation` altına yazılır.
- Komut satırı: `--rinex` + PPP-AR tekniği + `--fcb-stations` (ve `--ar-products` yok) aynı akışı çalıştırır; `--rinex` olmadan ya da PPP tekniğiyle yalnız FCB kestirimi yapılır.
- Farklı kod izleyen istasyonlar (örneğin P1 yerine C1C, LAMA) MW'yi tutarsız yapar; `--code-biases` ile verilen Bias-SINEX DSB/OSB dosyası bunları C1W/C2W'ye hizalar. Arayüz, OSB ürününden farklı merkezin `.BIA` dosyasını bunun için otomatik seçer.
- Çıktılara ayrıca `wide_lane_ambiguities.csv`, `narrow_lane_ambiguities_<yöntem>.csv` ve WL için de dışarıda bırakmalı doğrulama eklenir.

## Kullanım

```bash
python3 -m gps_processing --sp3-day Data/fcb/post_2023327/GBM0MGXRAP_20233270000_01D_05M_ORB.SP3 \
  --clock-file Data/fcb/post_2023327/GBM0MGXRAP_20233270000_01D_30S_CLK.CLK \
  --antex Data/fcb/post_2023327/igs20_2287.atx \
  --fcb-stations Data/net/STA1.rnx Data/net/STA2.rnx Data/net/STA3.rnx Data/net/STA4.rnx Data/net/STA5.rnx \
  --fcb-method all --output results
```

Arayüzde: *Processing settings → FCB estimation from a reference network*, istasyonları ekleyip **Estimate FCB**.

## Sınırlar

- NL FCB'leri yalnız kestirimde kullanılan yörünge/saat ürünüyle geçerlidir; dosya başlığında hangi ürün kullanıldığı yazılır.
- NL için float PPP'nin yakınsamış olması gerekir (σ_NL < 0.2 çevrim, birkaç saatlik veri). Bu yüzden 12 dakikalık örnek RINEX ile NL üretilemez.
- **Gerçek ağ denemesi (23.11.2023; IISC, LAMA, MYRA, NOVM, NRC1; GBM SP3/CLK, igs20 ANTEX, WHU kod biasları; gelgit açık):**
  - WL: 479 arc'ın 469'u sabitlendi; dışarıda bırakılan istasyonda kalan kesir RMS 0.11 çevrim, %87'si 0.15 altında. WL çalışıyor.
  - NL: dışarıda bırakılan istasyonda RMS 0.22–0.23 çevrim (rastgele: 0.29), 0.15 altı %49–51. EKF+ML en iyi (0.217) ama fark küçük; NL FCB'leri zayıf.
  - NOVM'de 245 kısa WL arc'ı var (diğerlerinde ~55): sık arc kesilmesi yüzünden NL'ye hiç girmedi. Ağ fiilen 4 istasyon ve kıtalara dağınık; aynı uyduyu aynı blokta gören istasyon sayısı az.
  - MADR PPP-AR/SDBS bu FCB'lerle: FCB'si olmayan uydular atıldığı için 1306 epokta 5'ten az uydu kaldı; fix oranı %1.5.
- **Asıl darboğaz kullanıcı tarafı NL:** Tutarlı GBM OSB ürünüyle bile MADR'da 2 saatten uzun arc'larda SD N1 float kesirlerinin medyanı 0.17 çevrim (WL'de 0.0); fix oranı %1.7, retler çoğunlukla N1 ratio testinden. IF fazda hâlâ ~1–2 cm düzeyinde modellenmemiş bir etki var (adaylar: tutulma dönemi yönelimi/OBX, okyanus yüklemesi, Shapiro, uydu PCV, stokastik model). Bu çözülmeden ağdan NL FCB kestirimi de iyileşmez.
- Float PPP'de MADR son konumu RINEX başlığına göre D 0.42 / K 0.34 / Y −0.05 m farklı. Yön ve oran Avrasya levha hareketine uyuyor (başlık koordinatı yaklaşık 20 yıllık olabilir); bağımsız bir IGS SINEX koordinatıyla doğrulanmadı.
