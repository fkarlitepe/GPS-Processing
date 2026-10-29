# Yer modelleri, uydu yönelimi ve dar şerit teşhisi (v2.9)

Bu belge, literatür değerlendirmesinde (PRIDE PPP-AR, Geng 2012/2019, Dai 2019, Ge 2008) eksik
olarak belirlenen modelleri ve dar şerit (NL) sorununu ayırt etmek için eklenen teşhis aracını
açıklar. Bütün düzeltmeler IF kod ve fazdan aynı miktarda çıkarılır (MW geometriden bağımsızdır).
Kapsam EKF ve PPP-AR çözümleridir; sıralı v2.0 temel çözümü değişmez.

## Data Files sekmesi

| Model | Girdi | Büyüklük | Kaynak |
|---|---|---|---|
| Katı yer gelgiti | yok (Ay/Güneş efemerisi) | radyal ±30 cm | IERS 2010 denk. 7.5 |
| Okyanus yüklemesi | BLQ dosyası (Onsala servisi, FES2014b, "CE frame") | kıyıya yakın birkaç cm, yarım günlük | IERS 2010 §7.1.2; 11 bileşen, ARG argümanları (RTKLIB `tide_oload`) |
| Kutup gelgiti | IGS ERP dosyası (Xpole, Ypole) | birkaç mm, gün içinde neredeyse sabit | IERS 2010 denk. 7.24, IERS 2018 doğrusal ortalama kutup |
| Shapiro gecikmesi | yok | ~1.3 cm (başucu) – ~1.9 cm (ufuk) | 2GM/c² ln((rs+rr+ρ)/(rs+rr−ρ)) |
| Uydu PCV | ANTEX (NOAZI, nadir açısı) | mm–cm | IGS ANTEX |
| Uydu yönelimi | ORBEX `*_ATT.OBX` (1. bölüm, merkez ürünü) | tutulma/öğle dönüşlerinde yaw 180°'ye kadar | GFZ/CODE kuaterniyonları |

BLQ için istasyon, RINEX `MARKER NAME` (tam ad, sonra ilk 4 karakter) ile eşlenir. ORBEX
kuaterniyonu ECEF vektörlerini gövde eksenine çevirir (r_gövde = R(q) r_ECEF); R'nin satırları
gövde eksenleridir. Bu kural GBM dosyasında doğrulandı: z ekseni yer merkezine 0.01° içinde, x
ekseni tutulma dışında nominal yaw ile 0.1° içinde.

## Ürün tutarlılığı: dosya adı değil başlık

`pipeline.product_consistency` merkezleri dosya **başlığından** okur: SP3 ajansı, CLK
`ANALYSIS CENTER`, Bias-SINEX dosya ajansı, ORBEX `CREATED_BY`. Kodlar tek isme indirgenir
(GBM/GFZ → GFZ, GRG/CNT → CNES, COD → CODE, WUM/WHU → WHU…).

**Bulgu (Data/fcb/post_2023327):** `GBM0MGXRAP_20233270000_01D_30S_ABS.BIA` adı GFZ olsa da
başlığı CNES'tir (`%=BIA 1.00 CNT`, "CNES OSB solutions", PPP-Wizard). Saat ve yörünge GFZ'nindir.
CNES faz biasları yalnız CNES (GRG) saatleriyle tamsayı özelliği taşır. Bu karışım artık hata
olarak durdurulur. MADR'da WL'nin çalışıp NL'nin hiç çözülememesinin nedeni budur (aşağıya bakın).

## Dar şerit teşhisi (`processing/diagnostics.py`)

1. Koordinat sabitlenir. Kullanıcı referansı varsa o kullanılır, yoksa aynı günün yakınsamış
   float PPP'sinin son saatinin ortalaması alınır.
2. AR filtresi tamsayı sabitlemeden çalıştırılır.
3. 30 dakikadan uzun ve σ < 0.05 çevrim olan yaylarda, tamsayı WL'ye koşullanmış SD N1 kesirleri
   15 dakikada bir örneklenir.
4. Kesirlerin dağılımı ve en kötü uydular raporlanır. Dağılım düzgün dağılıma benziyorsa ürünler
   uyumsuzdur ya da cm düzeyinde bir model hatası vardır; sıfırda kümeleniyorsa ürünler
   kullanılabilir demektir.

Kullanım: `--diagnose-nl` (CLI) veya arayüzde AR bölümündeki **Narrow-lane diagnostic**. Varsayılan
olarak seçili modellerle ve ek modeller (okyanus/kutup/Shapiro/yönelim/uydu PCV) kapalıyken iki
varyant karşılaştırılır.

### MADR, 23.11.2023 sonuçları

| Ürün seti | Örnek | \|kesir\| medyan | ±0.10 içinde | ±0.15 içinde |
|---|---|---|---|---|
| GBM SP3/CLK + CNES OSB, tüm modeller | 266 | 0.297 | %17 | %21 |
| Aynı, ek modeller kapalı | 274 | 0.290 | %20 | %26 |
| GBM SP3/CLK + JXF FCB, tüm modeller | 342 | 0.265 | %21 | %27 |
| Düzgün (rastgele) dağılım | — | 0.250 | %20 | %30 |

Sağlıklı bir sistemde (Dai 2019, Ge 2008) %90 civarı ±0.1 çevrim içindedir. Üç ürün seti de
rastgele dağılımdan ayırt edilemiyor ve modelleri kapatıp açmak sonucu değiştirmiyor. Yani NL
sorunu model eksikliğinden değil, ürün uyumsuzluğundan kaynaklanıyor. Eklenen modeller doğru ve
gerekli, ama bu veri setinde tek başlarına NL'yi kurtaramazlar.

Karşılaştırma için GBM yörünge/saatiyle aynı gün kestirilen **ağ FCB'si** (5 istasyon, EKF, tanımı gereği tutarlı ama zayıf ağ: LOO RMS 0.22 çevrim) ile:

| Ürün seti | Örnek | \|kesir\| medyan | ±0.10 içinde | ±0.15 içinde |
|---|---|---|---|---|
| GBM + ağ FCB, tüm modeller | 270 | 0.237 | %26 | %40 |
| GBM + ağ FCB, ek modeller kapalı | 274 | 0.261 | %26 | %35 |

Tutarlı ürünle dağılım rastgele dağılımdan ayrılmaya başlıyor ve ek modeller ±0.15 oranını 5 puan artırıyor. Ama 5 seyrek küresel istasyon NL için yetersiz kalıyor.

**Gerekli veri:** aynı gün için aynı merkezin tam seti. Örnek olarak GBM SP3 + CLK + GFZ'nin kendi
faz OSB'si ve `ATT.OBX`. Ya da GRG (CNES) SP3 + CLK + CNES OSB. Ya da WUM SP3 + CLK + `WUM…OSB.BIA`.
