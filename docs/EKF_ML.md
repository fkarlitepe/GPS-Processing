# EKF ve Makine Öğrenmesi ile PPP / PPP-AR kestirimi

Bu belge v2.1'de eklenen kestirim katmanını açıklar: iterasyonlu genişletilmiş Kalman filtresi (EKF), LAMBDA tabanlı kısmi belirsizlik çözümü (PAR) ve iki makine öğrenmesi bileşeni. Kodlar SDBS (Collins 2008; Shi 2012 §4.1) ve DCM (Collins 2008; Shi 2012 Bölüm 3) kullanıcı modellerinin **üzerine** kuruludur; ürün sözleşmesi ve işaret tanımları için [PPP_AR.md](PPP_AR.md).

## Kestirimin nerede yapıldığı ve ne eklendi

| Kestirim noktası | v2.0 | v2.1 | Modül |
|---|---|---|---|
| Konum/saat/belirsizlik durumu | Süreç gürültüsüz sıralı ağırlıklı LS | İterasyonlu EKF: süreç gürültüsü, statik/kinematik, Joseph formu | `processing/estimation/ekf.py`, `ppp_ekf.py`, `ar/solver.py` |
| Troposfer | Yok | Saastamoinen ZHD + Niell (veya Chao) eşleme, kalan ZWD rastgele yürüyüş durumu | `processing/troposphere.py` |
| Uydu saati | SP3 | SP3 + periyodik relativistik terim −2 r·v / c | `processing/orbit.py` |
| Aykırı gözlem | Yok | IGG-III eşdeğer ağırlık (sonsal standart artık) | `ekf.py: igg3` |
| Model uyumsuzluğu | Yok | İnovasyon tabanlı adaptif fading (yalnız konum) | `ekf.py: update(adaptive=…)` |
| Kayma tespiti | LLI, boşluk, MW | + Geometry-free faz sıçraması | `processing/cycle_slip.py` |
| Gözlem varyansı | Sabit σ/sin(e) | **ML:** artıklardan öğrenilen σ(e, SNR, yay) | `processing/ml/stochastic.py` |
| Tamsayı arama | Sınırlı ILS, ≤4 uydu | LAMBDA dekorelasyonu + bootstrapping PAR | `processing/ar/integer.py` |
| Fix kabulü | Ratio ≥ 3 | Ratio, **ML** doğrulayıcı veya ikisi | `processing/ml/ar_validation.py` |

## 1. Genişletilmiş Kalman filtresi

Durum vektörü (float PPP): `x = [X Y Z, c·dt_r, ZWD, A_1 … A_32]`; PPP-AR'da SDBS için `[XYZ, ZWD, (N1, NW)_1…32]`, DCM için ek olarak üç alıcı saat/bias durumu.

Zaman güncellemesi `x⁻ = x⁺`, `P⁻ = P⁺ + Q·Δt`:

- Statik konum: `Q_xyz = 0`; kinematik: `Q_xyz = q_p² Δt` (q_p m/√s).
- Alıcı saati: her epok beyaz gürültü (yeniden başlatma, 1e10 m²).
- ZWD: `Q = (q_T/1000)² Δt/3600` (q_T mm/√saat, varsayılan 5).
- Belirsizlikler sabit; kayma veya referans değişiminde sıfırlanır.

Ölçüm modeli (metre, iyonosferden bağımsız):

```
P_IF = ρ + c·dt_r − c·dt^s − δ_rel + ZHD·m_h(e) + ZWD·m_w(e) + ε_P
L_IF = ρ + c·dt_r − c·dt^s − δ_rel + ZHD·m_h(e) + ZWD·m_w(e) + A + ε_L
δ_rel = −2 (r·v)/c
```

Ölçüm güncellemesi Gauss–Newton iterasyonlu EKF'dir (IEKF): `h(x)` her iterasyonda yeniden doğrusallaştırılır, kazanç öncül kovaryansla hesaplanır, kovaryans Joseph formuyla güncellenir. Çıktıdaki `nis` normalleştirilmiş inovasyon karesidir (gözlem başına, beklenen değer 1): 1'in çok üstü stokastik modelin iyimser olduğunu gösterir.

**Robust (IGG-III):** sonsal artıklar kendi kovaryansları `R − H P⁺ Hᵀ` ile standartlaştırılır; `|v| ≤ 2.5` tam ağırlık, `2.5 < |v| ≤ 6` azaltılmış ağırlık, `> 6` reddedilir. SDBS'de satırlar korele olduğundan ağırlık uydu bloğuna (kod+faz+MW) uygulanır.

**Tek EKF çekirdeği (v2.9, `processing/engine.py`):** Float PPP ve üç PPP-AR kullanıcı modeli aynı döngüyü kullanır. Bu döngü şunları bir kez yapar: durum düzeni, yükseklik maskesi, öngörü, saat ve kayma sıfırlamaları, atmosfer gecikmeleri, uydu saat/relativite terimleri, robust/adaptif güncelleme. Gözlem modelleri eklentidir: `ppp_ekf.IonosphereFreeModel` (IF kod+faz) ve `ar.solver.UserModel` (IF kod, IF faz, MW; SDBS/DCM/IPC dönüşümü; güncellemeden sonra AR). Yeniden düzenleme v2.8 sonuçlarını bit düzeyinde aynen üretir (MADR, 4 saat; PPP, SDBS, DCM, IPC). DCM/IPC başlangıç epoklarındaki koşul sayısı o kadar kötüdür ki, gürültü matrisinin işlem sırası bile son basamaktaki farkları desimetrelere büyütebiliyor; vektörleştirilmiş kodda işlem sırası bu yüzden korunur.

**Adaptif EKF (v2.9):**

- *Gözlem grubuna göre varyans faktörleri:* kod / faz / MW × GPS / Galileo için ayrı faktör. İnovasyonlardan Mehra / Sage–Husa biçiminde kestirilir ve epoklar boyunca yumuşatılır. Alıcı saati (beyaz gürültü) bir grubun bütün satırlarında ortak olduğundan inovasyonlar grup içinde merkezlenir; bu, saati tam olarak yok eder. Yeni belirsizlikli satırlar dışarıda tutulur. Bu, Guo & Zhang (2014)'ün "gelecek çalışma" olarak bıraktığı çoklu uyarlamalı faktördür.
  - Sonsal artıklarla yapılan ilk deneme filtreyi kendi çıktısına bağladı ve varyanslar alt sınıra çöktü (4 saatte 1 m). Bu yüzden kullanılmıyor.
- *Fading* yalnız kinematik alıcıda ve yalnız anlamlı uyumsuzlukta (NIS > χ²₀.₉₉₉(m)/m) uygulanır. Statik konumda dinamik model hatası olmaz; eski sabit 1.5 eşiği MADR'da statik çözümü desimetrelerce kaydırıyordu.

MADR 23.11.2023 (GPS+Galileo, GBM SP3/CLK, igs20) float PPP, 3B hata (m), referans aynı günün yakınsamış koordinatı:

| | 15 dk | 30 dk | 60 dk | 4 sa | 8 sa | 23 sa |
|---|---|---|---|---|---|---|
| EKF | 0.090 | 0.186 | 0.023 | 0.025 | 0.017 | 0.012 |
| Eski adaptif (sabit eşikli fading) | 0.067 | 0.223 | 0.024 | 0.286 | 0.901 | 0.584 |
| v2.9 adaptif (grup faktörleri, statikte fading yok) | 0.066 | 0.166 | 0.024 | 0.022 | 0.023 | 0.022 |

Grup faktörleri ilk yarım saati iyileştiriyor, sonrasında yaklaşık nötr (1 cm kötü). Fading'in statik moddan çıkarılması büyük hatayı gideriyor.

## 2. LAMBDA ve kısmi belirsizlik çözümü (PAR)

1. Float belirsizlik kovaryansı `Q = LᵀDL` biçiminde ayrıştırılır, tamsayı Gauss dönüşümleri ve permütasyonlarla dekorele edilir (`Z` unimodüler, `Q_z = ZᵀQZ`).
2. Bootstrapping başarı oranı `P_s = Π (2Φ(1/(2√d_i)) − 1)` hesaplanır (Teunissen 1998; ILS başarı oranının alt sınırı).
3. En küçük koşullu varyanslı bileşenlerden başlayarak `P_s ≥ P_0` (varsayılan 0.999) olan en büyük alt küme seçilir.
4. Alt kümede tam ILS araması yapılır, ratio ve/veya ML testi uygulanır.
5. Kabul edilen kombinasyonlar `z = Zᵀa` üzerinden tüm duruma koşullandırılır (`condition_linear`). Önce WL, sonra koşullu N1 için tekrarlanır.

PAR kombinasyon sabitlediğinden, kısmi çözümlerde tekil N1 değerleri tamsayı olmayabilir; `n3_amb` yalnız tamsayıya koşullanmış değerleri gösterir. Tüm adaylar sabitlenirse durum `fixed`, alt kümede `partial_fixed`'tir. `ar_diagnostics` WL/N1 ratio, başarı oranı ve ML olasılığını içerir.

#### LAMBDA'ya alternatif tamsayı kestiricileri (Teunissen, Joosten & Tiberius 2002)

Makalenin sınıflaması üç eksene dayanır: **kestirici** (yuvarlama IR, bootstrapping IB, integer least squares ILS), **dönüşüm** (orijinal, frekansa göre önceden tanımlı, LAMBDA dekorelasyonu) ve **model** (geometriden bağımsız, geometriye dayalı). TCAR (Galileo E1/E2/E4) ve CIR (GPS L1/L2/L5), önceden tanımlı dönüşümle yapılan geometriden bağımsız bootstrapping'tir. LAMBDA ise ILS'dir ve başarı oranı en yüksek kestiricidir (Teunissen 1999).

`integer_method` seçenekleri (arayüz: *Integer method*, CLI: `--integer-method`):

| Seçenek | Kestirici | Dönüşüm | Kısmi AR | Başarı oranı (`success`) |
|---|---|---|---|---|
| `lambda` | ILS (arama) | LAMBDA dekorelasyonu | Var | Kesin IB, dekorelasyonlu: ILS alt sınırı |
| `ils` | ILS, en fazla 4 uydu, ya hepsi ya hiçbiri | Arama için dekorelasyon (sonucu değiştirmez) | Yok | Kesin IB, dekorelasyonlu |
| `bootstrap` | IB (arama yok, sabit adım sayısı) | LAMBDA dekorelasyonu | Var | Kesin IB, aynı zamanda gerçek başarı oranı |
| `cascade` | Her uydu çifti ayrı yuvarlanır | Önceden tanımlı: WL → N1 (TCAR/CIR ilkesi) | En hassas uydular | IR alt sınırı, Ps-LAMBDA denk. 26 |

- **Çift frekansta TCAR = CIR:** L1/L2'de önceden tanımlı dönüşüm yalnızca `N_W = N1 − N2` ve ardından `N1`'dir. Çözücünün durum vektörü zaten bu kombinasyonları kullanır ve WL'den sonra N1'i koşullar. Bu yüzden `cascade` seçeneği, frekanslar arasında bootstrapping (önce WL, sonra N1) ile uydular arasında korelasyonsuz yuvarlamayı birleştirir. Üç frekanslı TCAR/CIR adımı (EWL) için üçüncü bir frekans gerekir; şu anki veri yalnızca L1/L2'dir.
- **Ratio:** `bootstrap` ve `cascade` arama yapmadığı için ikinci aday, çözümün 2k birim komşusu arasında en küçük normlu olandır. Ratio testi ve ML doğrulayıcı bu yöntemlerle de çalışır. ML doğrulayıcı LAMBDA sonuçlarıyla eğitildiğinden, diğer yöntemlerde ratio veya başarı oranı ile kullanılması önerilir.
- **Makaleyle doğrulama** (`tests/test_integer_methods.py`): Tablo 3 ve 4'teki koşullu σ değerlerinden başarı oranları (0.270 / 0.906 / 0.914 ve 0.287 / 0.978 / 0.980) ve Bölüm 5.3'teki kısmi fix oranları (0.8586, 0.7540, 0.6330, 0.4287) yeniden üretildi. TCAR ve CIR dönüşümlerinin admissible olduğu (det = ±1) ve birim alt üçgen dönüşümlerin IB başarı oranını değiştirmediği gösterildi. Simülasyonda P(cascade) ≤ P(IB) ≤ P(ILS) sıralaması sağlandı.
- **Sentetik PPP-AR verisinde:** N1 başarı oranı `lambda` ve `bootstrap` ile ≈ 1.00, `cascade` ile ≈ 0.84 çıktı. `cascade`, 0.99 eşiğinde hiç fix vermedi, 0.8 eşiğinde verdi. Bu, makalenin geometriye dayalı model sonucuyla (Şekil 3–4) uyumlu: uydu çiftleri arasındaki korelasyonu yok saymak başarı oranını düşürür.

Karşılaştırma sekmesine **AR · Bootstrapping (decorrelated)** ve **AR · Cascade rounding (TCAR/CIR)** varyantları eklendi.

#### Başarı oranı sınırları (Ps-LAMBDA, Verhagen, Li & Teunissen 2013)

`processing/ar/integer.py` makaledeki kapalı formları uygular. Denklem numaraları makaleye aittir:

| Büyüklük | Denklem | Kodda | Kullanım |
|---|---|---|---|
| Kesin IB başarı oranı (dekorelasyonlu) | (28), (29), (32) | `success`, `success_rates()['ib']` | ILS alt sınırı; PAR alt kümesi ve kabul kapısı |
| IB, orijinal sıralama | (28) | `success_rates()['ib_original']` | Yalnız karşılaştırma için; dekorelasyonsuz değer çok kötümser |
| IR alt sınırı | (26) | `ir_lower` | Tanı |
| ADOP | (31) | `adop` | ML doğrulayıcı özniteliği, tanı |
| ADOP yaklaşıklığı / IB üst sınırı | (30) | `adop_approx` | Tanı |
| ADOP tabanlı ILS üst sınırı | (33) | `success_upper`, `ils_upper` | Tanı: üst sınır eşiğin altındaysa AR başarılı olamaz |
| Özdeğer sınırları | (36)–(37) | `eig_lower`, `eig_upper` | Tanı (makalede de gevşek) |
| Simülasyon tabanlı başarı oranı | (25) | `simulate_success()` | Yalnız çevrimdışı analiz; makale ≥10⁵ örnek önerir |

`ar_diagnostics` her epok için `wl/n1_success` (alt sınır), `wl/n1_success_upper` ve `wl/n1_adop` değerlerini yazar. Makaledeki örnek kovaryans (denklem 38) için testler IR ≤ IB ≤ ILS ≤ ADOP üst sınırı sıralamasını ve IB formülünün kesinliğini doğrular (`tests/test_success_rate.py`). Bölgeyi sınırlayan ILS sınırları (34)–(35) uygulanmadı. Makaleye göre alt sınır gevşek, üst sınır da hesaplama açısından ağırdır.


## 3. Makine öğrenmesi bileşenleri

Ortamda scikit-learn bulunmadığı ve sonuçların JSON olarak saklanabilmesi istendiği için iki öğrenici numpy ile yazıldı (`processing/ml/learners.py`): gradyan artırmalı regresyon ağaçları (Friedman 2001) ve L2 düzenlileştirilmiş lojistik regresyon (IRLS).

### 3.1 Öğrenilmiş stokastik model (gözlem ağırlıklandırma)

Kalman filtresinin performansı `R` matrisinin doğruluğuna bağlıdır. Sabit `σ/sin(e)` modeli alıcı, anten ve çok yol ortamını yansıtmaz.

1. Öğrenme geçişi: float EKF PPP varsayılan ağırlıklarla çalıştırılır, her gözlem için sonsal kod/faz artığı saklanır (`residuals.csv`).
2. Özellikler: `sin(e)`, L1/L2 SNR (dB-Hz), `log(1 + yay uzunluğu)`.
3. Hedef: `log(v²)`; `E[log χ²₁] = −1.27` düzeltmesiyle varyansa dönüştürülür. Kod ve faz için ayrı GBRT modelleri.
4. Asıl çözüm (float veya AR) öğrenilen σ'larla tekrarlanır; σ'lar varsayılan modelin 0.3–30 katı arasında sınırlandırılır.

Bu, varyans bileşeni kestirimi (VCE) fikrinin özellik tabanlı, doğrusal olmayan bir genellemesidir. Sınırı: model aynı oturumun artıklarından öğrenir; bağımsız doğrulama için bir istasyon/günde eğitip başka bir günde uygulamak gerekir (`ml_models.json` modeli saklar).

### 3.2 ML ambiguity doğrulayıcısı (v2.3, scikit-learn)

Sabit ratio eşiği bir başarısızlık oranı garantisi vermez (Teunissen & Verhagen). Doğrulayıcı, bir tamsayı aşamasının (WL veya N1) sonucundan "fix doğru mu?" olasılığını kestirir.

- **Eğitim verisi** (`processing/ml/ar_simulation.py`): float kovaryanslar PPP-AR kullanıcı modelinden üretilir. 5–10 uydu, rastgele gökyüzü, uydular arası tek fark. WL aşaması için MW ortalaması (multipath nedeniyle azaltılmış etkin epok), N1 aşaması için konum + ZWD + N1 ile IF kod/faz en küçük kareleri (1–120 epok, 30 s). Filtrenin bilmediği iki hata eklenir: kovaryans ölçek hatası `κ·Q`, log κ ~ N(0, 0.4), ve örneklerin %20'sinde 1–2 belirsizlikte ±0.25 çevrime kadar modellenmemiş bias. Etiket, sabitlenen kombinasyonların gerçek tamsayılara eşit olup olmadığıdır; bu yüzden kısmi fix'ler de doğru etiketlenir.
- **Özellikler (9):** log ratio, logit Ps (IB alt sınırı), logit ADOP tabanlı üst sınır, log ADOP, boyut, sabitlenen oran k/n, log(1 + en iyi norm/k), log(1 + ikinci norm/k), aşama (WL/N1).
- **Olasılık kalibrasyonu:** model ayrı bir örnekle izotonik regresyonla kalibre edilir. Böylece `ml_threshold = 0.99` yaklaşık olarak "doğru fix olasılığı ≥ %99" anlamına gelir.
- **Çözücüde:** `ar_validation = ml` veya `ratio+ml` iken başarı oranı kapısı uygulanmaz, çünkü Ps zaten modelin girdisidir. PAR alt kümesi yine `ar_success_rate` ile seçilir.

#### Model seçimi: sabit hata oranında karşılaştırma

`python3 -m gps_processing.processing.ml.ar_benchmark` altı aday modeli ve klasik kuralları karşılaştırır. Her kuralın eşiği kalibrasyon kümesinde hedef hata oranına (kabul edilenler içinde yanlış fix oranı) göre seçilir, sonra bağımsız test kümesine uygulanır. Ölçüt **kullanılabilirliktir**: doğru kabul edilen örnekler / tüm örnekler. Her tekrarda 12 000 eğitim, 8 000 kalibrasyon ve 8 000 test örneği vardır; sonuçlar 3 tekrarın ortalamasıdır (`results/ar_benchmark/`).

| Kural | Kullanılabilirlik @ %0.3 hedef | Gerçekleşen hata | Kullanılabilirlik @ %1 hedef | Gerçekleşen hata | AUC |
|---|---|---|---|---|---|
| **Lojistik regresyon, 2. derece (sklearn)** | **0.269 ± 0.009** | 0.37 % | **0.322 ± 0.006** | 1.06 % | 0.920 |
| Sinir ağı (MLP 32-16) | 0.247 ± 0.021 | 0.37 % | 0.307 ± 0.015 | 1.10 % | 0.918 |
| Gradyan artırma (HistGB) | 0.245 ± 0.010 | 0.42 % | 0.316 ± 0.004 | 1.08 % | 0.917 |
| Random forest | 0.225 ± 0.019 | 0.36 % | 0.299 ± 0.004 | 1.05 % | 0.911 |
| Lojistik regresyon (numpy, v2.1) | 0.150 ± 0.018 | 0.33 % | 0.246 ± 0.013 | 1.08 % | 0.913 |
| SVM (RBF) | 0.038 ± 0.014 | 0.36 % | 0.134 ± 0.053 | 1.25 % | 0.909 |
| Ps, sabit hata oranına kalibre | 0.043 | 1.35 % | 0.046 | 1.25 % | — |
| Ratio, sabit hata oranına kalibre | 0.000 | — | 0.000 | — | — |
| Ps ≥ 0.999 + ratio ≥ 3 (önceki varsayılan) | 0.135 | 0.12 % | 0.135 | 0.12 % | — |
| Ratio ≥ 3 (sabit) | 0.475 | **24.4 %** | 0.475 | **24.4 %** | — |

Yorum:
- **Sabit ratio ≥ 3**, bias ve kovaryans hatası olan bu senaryoda her dört fix'ten birini yanlış kabul ediyor. Hiçbir ratio eşiği %1 hatayı sağlayamıyor: en yüksek ratio'lar bile, başarı oranı düşükken zorlanan 1 boyutlu kısmi fix'lerde yanlış olabiliyor.
- **Ps ≥ 0.999 + ratio ≥ 3** güvenli (%0.12 hata) ama tutucu: doğru fix'lerin %78'ini reddediyor.
- **Seçilen model** (2. derece lojistik regresyon) %0.37 gerçekleşen hata ile **iki kat** kullanılabilirlik sağlıyor (0.269'a karşı 0.135). İki hedefte de birinci, tekrarlar arası değişkenliği en düşük ve tahmin süresi örnek başına ~2 µs. Bu yüzden `DEFAULT_MODEL` budur.
- Daha karmaşık modeller (HistGB, MLP, RF) daha iyi değil. Öznitelikler az ve düzgün; ikinci derece etkileşimler (ör. ratio × Ps) karar sınırını yakalamaya yetiyor.
- Gerçekleşen hata, hedefi biraz aşabiliyor (%0.37'ye karşı %0.30), çünkü eşik sonlu bir kalibrasyon kümesinden seçiliyor. Güvenlik için hedef biraz daha düşük seçilebilir.

**Sınır:** Bu sonuçlar simülasyona dayanıyor. Gerçek ölçümlerdeki bias dağılımı farklı olabilir; gerçek AR ürünüyle işlenmiş günlerde fix doğruluğu (ör. bilinen koordinata göre) ayrıca doğrulanmalıdır.

Arayüzdeki **Train validator** düğmesi seçilen modeli 8000 örnekle eğitip `Data/models/ar_validator_<model>.joblib` olarak kaydeder. Dosya seçilmezse varsayılan model ilk kullanımda 4000 örnekle bellekte eğitilir (deterministik tohum, ~10 s). v2.1'deki JSON doğrulayıcılar eski öznitelik kümesini kullandığı için yeniden eğitilmelidir. Uydu yörünge ve saat enterpolasyonu değişmedi: 9. derece Lagrange.

### 3.3 Bias'a duyarlı ML doğrulayıcı (v2.9)

Doğrulayıcının öznitelikleri genişletildi. Artık FFRT eşiğine olan pay (log(ratio/μ)), sabitlenen kümedeki en küçük Dong–Bock P0 ve en büyük kesirli kısım da giriyor (`ml/ar_simulation.py`). Monte Carlo verisi kovaryans ölçek hatası ve modellenmemiş kesirli biaslar içerdiği için, formal testlerin göremediği biası bu öznitelikler yakalar. `ar_benchmark` aynı sabit hata oranında FFRT, FFRT+P0, kalibre P0 ve ML'yi karşılaştırır. Kayıtlı eski doğrulayıcılar farklı öznitelik kümesi nedeniyle reddedilir; yeniden eğitin.

### 3.4 Kendi kendini eğiten çözülebilirlik modeli (v2.9, `ml/fixability.py`)

1. AR filtresi önce tamsayı sabitlemeden çalıştırılır.
2. Sonunda yakınsamış (σ < 0.05, kesir < 0.10) her yayın tamsayısı geriye dönük olarak bilinir.
3. Bu yayın önceki her epoğu "şimdi yuvarlasam doğru olur muydu" sorusuyla etiketlenir.
4. Yakınsadığı hâlde tamsayıya oturmayan yay (ürün/model hatası) "çözülemez" olarak etiketlenir.
5. GBRT, gerçek zamanda bilinenlerden bu olasılığı öğrenir: σ, kesir, P0, yükseklik, yay uzunluğu ve aşama. Model uyduya göre çapraz eğitilir; hiçbir uydu kendi etiketini görmüş bir modelle puanlanmaz.

İkinci geçişte bu olasılık adayları sıralar, eşiğin (0.5) altındakileri dışlar ve aramayı başlatır: en az `ar_min_ambiguities` aday çözülebilir görünmeden arama yapılmaz. Bu, PPP-ARISEN'in CSI-ZWDV yakınsama göstergesinin veriye dayalı karşılığıdır. Seçenek: `--ar-fixability` / arayüzde *ML fixability*.

### 3.5 FCB EKF'sinde veriden öğrenilen süreç gürültüsü

`ekf+ml` FCB kestiricisi ilk EKF geçişinden her uydunun UFO'sunun bloklar arası değişiminin dayanıklı (MAD) varyansını alır, formal varyansı çıkarır ve ikinci geçişte uyduya özgü rastgele yürüyüş gürültüsü olarak kullanır (sınırlar 0.002–0.2 çevrim/blok). Ge'nin 15 dakikalık tabloları ile Geng'in "günlük yeterli" sonucu arasında veri karar verir. 5 istasyonlu ağda bütün uydular alt sınırda kaldı (0.004 çevrim/√saat): bu günde UFO'lar kararlı.

## 4. Örnek veride elde edilen sonuçlar

MADR, 23.11.2023, 30 s, **yalnız 25 epok (~12 dakika)**, IGS final SP3, 10° maske, referans: RINEX başlık koordinatı. Uyumlu AR ürünü olmadığından yalnız float PPP karşılaştırılabildi:

| Yöntem | Son 3B hata (m) | RMS D / K / Y (m, son yarı) | Ort. NIS |
|---|---|---|---|
| Sıralı LS (v2.0) | 11.48 | 9.34 / 3.04 / 6.22 | — |
| EKF (troposfer + relativite) | 1.47 | 0.74 / 0.97 / 1.34 | 3.42 |
| Adaptif + robust EKF | 2.26 | 1.52 / 0.83 / 1.78 | 0.49 |
| EKF + ML ağırlık | 0.90 | 0.97 / 0.46 / 0.48 | 0.87 |

Yorum:

- İyileşmenin büyük kısmı EKF'nin kendisinden değil, **fiziksel modellerden** (≈2.4 m troposfer ZHD, ±15 m'ye varan relativistik terim) gelir. EKF bunları kestirim çerçevesine yerleştirmeyi ve ZWD'yi durum olarak taşımayı sağlar.
- ML ağırlıklandırma NIS'i 3.4'ten 0.87'ye indirir: filtrenin belirsizlik beyanı gerçek hatalarla tutarlı hale gelir. AR için bu kritiktir çünkü başarı oranı ve ratio doğrudan `Q`'ya bağlıdır.
- Adaptif fading (NIS 0.49: kovaryans fazla şişirilmiş), modellenmemiş sistematik hata varken (15 dakikalık SP3 saatlerinin doğrusal enterpolasyonu, anten/wind-up eksikliği) sürekli tetiklenip sonucu kötüleştirdi. Kinematik veya kısa süreli bozucu olaylarda faydalıdır.
- 12 dakikada hiçbir yöntem 10 cm yakınsamaya ulaşmaz; float PPP tipik olarak 20–40 dakika, PPP-AR 10–20 dakika ister. Anlamlı yakınsama/fix oranı karşılaştırması için en az 2–6 saatlik veri ve uyumlu saat/bias ürünü gerekir.

## 5. Önerilen deney tasarımı (tez için)

1. Birden fazla IGS istasyonu, en az 7 gün, 24 saatlik 30 s veri; günlük oturumları 2'şer saatlik yaylara bölün.
2. Her yay için Karşılaştırma sekmesinde: Sıralı LS, EKF, Adaptif+robust EKF, EKF+ML ağırlık; PPP-AR'da ILS+ratio, LAMBDA+PAR, LAMBDA+ML.
3. Referans: IGS SINEX haftalık koordinatları (`--reference-xyz`).
4. Metrikler (`comparison/statistics.csv`): yakınsama süresi (yatay 10 cm / düşey 20 cm), son yarı RMS, fix oranı, yanlış fix oranı (referansa göre >5 cm sapan fixed epoklar), ortalama NIS.
5. ML ağırlık modelini bir günde eğitip diğer günlere uygulayarak (çapraz doğrulama) genellenebilirliği test edin.

## 6. Sınırlar

Modeller v2.9 itibarıyla tamdır: ANTEX PCO/PCV, uydu PCV, ORBEX yönelim, wind-up, katı yer/okyanus/kutup gelgiti, Shapiro, OSB/FCB, 30 s saat. Dar şerit sonucu ise ürünlerin aynı merkezden gelmesine bağlıdır (docs/MODELS.md). ML bileşenleri araştırma aracıdır; yenilik iddiası öncesinde güncel literatür taranmalı ve sonuçlar bağımsız günlerle/istasyonlarla doğrulanmalıdır.

## Kaynaklar

- Collins, P. et al. (2008). *Precise Point Positioning with Ambiguity Resolution using the Decoupled Clock Model*. ION GNSS.
- Shi, J. (2012). *Precise Point Positioning Integer Ambiguity Resolution with Decoupled Clocks*. PhD, University of Calgary.
- Teunissen, P.J.G. (1995). The least-squares ambiguity decorrelation adjustment. *J. Geodesy* 70.
- De Jonge, P., Tiberius, C. (1996). *The LAMBDA method for integer ambiguity estimation: implementation aspects*. LGR-Series 12, TU Delft.
- Teunissen, P.J.G. (1998). Success probability of integer GPS ambiguity rounding and bootstrapping. *J. Geodesy* 72.
- Teunissen, P.J.G., Joosten, P., Tiberius, C. (1999). Geometry-free ambiguity success rates in case of partial fixing. ION NTM.
- Verhagen, S., Teunissen, P.J.G. (2013). The ratio test for future GNSS ambiguity resolution. *GPS Solutions* 17.
- Yang, Y., Song, L., Xu, T. (2002). Robust estimator for correlated observations based on bifactor equivalent weights. *J. Geodesy* 76 (IGG-III).
- Yang, Y., He, H., Xu, G. (2001). Adaptively robust filtering for kinematic geodetic positioning. *J. Geodesy* 75.
- Saastamoinen, J. (1972). Atmospheric correction for the troposphere and stratosphere in radio ranging of satellites. AGU Monograph 15.
- Friedman, J.H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics* 29.
