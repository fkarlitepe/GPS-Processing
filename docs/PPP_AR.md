# PPP-AR: SDBS, Decoupled Clock Model ve Integer Phase Clock

Bu sürüm GPS L1/L2 için iki **deneysel statik kullanıcı çözümü** içerir. Ağdan FCB kestirimi, üretim seviyesinde hassas PPP ve gerçek ürünlerle doğrulanmış santimetre doğruluğu iddia edilmez. Mevcut depo verileri yalnız float PPP denemesi için yeterlidir; AR ürünleri depoda yoktur.

## Literatür ve uygulama karşılığı

- Shi, J. (2012), *Precise Point Positioning Integer Ambiguity Resolution with Decoupled Clocks*, University of Calgary. Bölüm 3, özellikle (3.15)–(3.17), kullanıcı datum tanımı §3.5; SDBS §4.1, (4.2)–(4.7). [Üniversite nüshası](https://www.ucalgary.ca/engo_webdocs/YG/12.20367_JunboShi.pdf).
- Collins et al. (2008), *Precise Point Positioning with Ambiguity Resolution using the Decoupled Clock Model*. [ION](https://www.ion.org/publications/abstract.cfm?articleID=8043).
- Ge et al. (2008), *Resolution of GPS carrier-phase ambiguities in Precise Point Positioning (PPP) with daily observations*. [DOI](https://doi.org/10.1007/s00190-007-0187-4).
- Teunissen & Verhagen, integer aperture / ratio kabulünün model bağımlılığı. [Birincil araştırma](https://gnss.curtin.edu.au/wp-content/uploads/sites/21/2016/04/Teunissen2007GNSS.pdf).
- Modern DCM genişletmesi: *An uncombined triple-frequency user implementation of the decoupled clock model for PPP-AR* (2021). [Makale](https://doi.org/10.1007/s00190-021-01510-y). Bu sürüm üç frekanslı modeli uygulamaz.

SDBS uygulaması tezdeki klasik önce IF float ardından bias düzeltme akışının birebir kopyası değildir: normalize uydu düzeltmelerinden sonra P_IF, L_IF ve MW üzerinde tek fark alıp N1/NW durumlarını ortak filtrede kestiren yeniden parametrelenmiş kullanıcı modelidir. DCM aynı gözlemleri fark almadan kullanır ve üç ayrı alıcı saat/bias durumu kestirir. Karşılaştırma sentetik verilerde yapılır; iki yöntemin gerçek veri performansı henüz ölçülmemiştir.

## Denklem, birim ve işaret

Kodda faz belirsizliği **pozitif** işaretlidir; Shi tezindeki negatif N tanımıyla N_code = -N_thesis dönüşümü gerekir. MW = L_WL - P_NL, NW = N1 - N2:

```
L_IF ambiguity (m) = lambda_N * N1 + beta * NW
lambda_N = c/(f1+f2)
beta = c*f2/(f1^2-f2^2)
MW ambiguity (m) = lambda_W * NW
lambda_W = c/(f1-f2)
```

Normalize edilmiş düzeltmeler gözleme **eklenir**:

```
P_IF + code_m  = rho + receiver_code_clock_m
L_IF + phase_m = rho + receiver_phase_clock_m + lambda_N*N1 + beta*NW
MW   + mw_m    = receiver_MW_bias_m + lambda_W*NW
```

DCM'de `code_m` ve `phase_m`, ilgili uydu saatinin c ile çarpılmış toplam düzeltmesidir; `mw_m` uydu MW bias düzeltmesidir. SDBS'de `code_m` kod saati düzeltmesi, `phase_m` bu saat artı uygun IF uydu faz/kod bias düzeltmesi, `mw_m` MW bias düzeltmesidir. İşaret sağlayıcının ürün tanımından dönüştürülmelidir. Sıradan CLK'yi faz saati olarak kopyalamak veya eksik biasları sıfırla doldurmak geçerli bir ürün dönüşümü değildir.

SDBS her üç gözlem türünde uydu eksi referans uydu farkını alır; alıcı ortak terimleri yok olur. DCM referans uydunun N1 ve NW değerlerini sıfıra bağlayıp alıcı saatlerini bununla uyumlu kestirir. Referans uydu kaybolduğunda veya kilit kaybettiğinde tüm belirsizlik datumu sıfırlanır. DCM saatleri bu datum altında tanımlıdır; mutlak fiziksel saat yorumu yapılmamalıdır.

## Ürün girişi

Şimdilik **native Bias-SINEX, FCB/UPD ve NRCan DC okuyucusu yoktur**. Sağlayıcı formatını ve işaret/datum tanımını bilen bir dönüştürücünün hazırladığı `togu-ar-1` JSON kabul edilir. Bu açık sınır farklı bias ürünlerinin yanlışlıkla birbirinin yerine kullanılmasını önler.

Şema örneği (tek satır yalnız biçimi gösterir, çalışır gerçek ürün değildir):

```json
{
  "schema": "togu-ar-1",
  "method": "SDBS",
  "date": "2023-11-23",
  "signals": ["C1W", "L1W", "C2W", "L2W"],
  "source": "Analiz merkezi / ürün serisi / dönüşüm açıklaması",
  "orbit_product": "IGS0OPSFIN_20233270000_01D_15M_ORB.SP3",
  "convention": "add_to_observation_m_positive_ambiguity",
  "records": [
    {"epok": 0, "sat": 1, "code_m": 123.4, "phase_m": 123.6, "mw_m": 0.12}
  ]
}
```

DCM için `method: DCM` yazılır ve üç düzeltme DCM ürünü tanımına göre hazırlanır. `epok` gözlem günü gece yarısından itibaren saniye, `sat` GPS PRN, düzeltmeler metredir. Tüm kullanılabilir çift frekanslı gözlem kayıtları için tam epok/uydu eşleşmesi gerekir. Okuyucu bias süreksizlikleri üzerinden enterpolasyon yapmaz. Tarih, yöntem, orbit dosya adı ve seçilmiş sinyaller kontrol edilir. Dosya adı eşleşmesi fiziksel ürün uyumluluğunu kanıtlamaz; sağlayıcı/datum uyumluluğu dönüşüm sırasında doğrulanmalıdır. Ürün belirsizlikleri henüz kovaryansa eklenmez.

## Kullanım

Arayüzde İşlem ayarları → PPP-AR/SDBS veya PPP-AR/DCM seçin ve AR saat/bias JSON dosyasını yükleyin. İki yöntem için açıklama sekmeleri vardır. CLI örneği:

```bash
python -m gps_processing \
  --sp3-before IGS0OPSFIN_20233260000_01D_15M_ORB.SP3 \
  --sp3-day IGS0OPSFIN_20233270000_01D_15M_ORB.SP3 \
  --sp3-after IGS0OPSFIN_20233280000_01D_15M_ORB.SP3 \
  --rinex MADR00ESP_R_20233270000_01D_30S_MO.rnx \
  --technique PPP-AR/SDBS --ar-products products/sdbs.json \
  --ar-min-epochs 20 --ar-ratio 3 --ar-max-std 0.15 \
  --output results/sdbs
```

## Kalite kontrolü ve tamsayı çözümü

- Ham P1/P2/L1/L2 bağımsız metre gürültüsü kombinasyonlara taşınır. IF ile MW korelasyonları ve tek farkların ortak referans kovaryansı korunur.
- Varsayılan ham kod/faz sigmaları 3 m / 3 mm; sin(yükseklik) ile ağırlıklandırılır. Bunlar alıcıya özgü kalibre edilmemiş varsayımlardır.
- Kilit kaybı, 120 s üzeri boşluk, sinyal değişimi veya 5 m MW sıçraması ilgili yayı sıfırlar. Referansın değişmesi tüm belirsizlikleri sıfırlar. Tam GF/istatistiksel cycle-slip tespiti henüz yoktur.
- En az 5 uydu; minimum yay uzunluğu ve standart sapma kapıları gerekir. Varsayılan 20 epok bir zaman garantisi değildir.
- Kısmi AR en fazla 4 uydu çifti üzerinde önce NW, ardından koşullandırılmış N1 için iki en iyi tamsayı adayını arar.
- Arama tam kovaryansla sınırlı sphere search'tür; **LAMBDA/MLAMBDA değildir**, dekorelasyon yoktur. Bütçe aşımı sabitlemeyi reddeder.
- İkinci/en iyi maliyet oranı ve son ölçüm artıkları kontrol edilir. Sabit ratio=3 eşiği bir güvenilirlik olasılığı veya sabit hata oranı garantisi değildir; FFRT henüz uygulanmadı.
- Kabul edilen koordinat düzeltmesi çapraz kovaryansla uygulanır. Fixed sonuç float filtreye geri beslenmez; yanlış sabitlemenin sonraki epoklara taşınması önlenir.

`xyz_result` float koordinatları, seçilen koordinatları ve `float/partial_fixed` durumunu içerir. `n3_amb` AR için N1/NW float/fixed alanlarını taşır; bunlar çevrimdir. `ar_diagnostics` WL/N1 ratio, referans, resetler ve reddetme nedenlerini içerir. `cll_result` ve `dx_1` **float** filtre durumuna aittir; AR boyutları SDBS için 67, DCM için 70'tir. DCM durum sırası XYZ, üç alıcı saat/biası (metre), GPS01…32 N1/NW (çevrim); SDBS'de üç saat durumu yoktur. SDBS alıcı saat çıktısı tanımsızdır.

## Üç modelin karşılaştırması (Shi & Gao 2014)

Shi & Gao (2014), *A comparison of three PPP integer ambiguity resolution methods* (GPS Solutions 18:519–528), SDBS, DCM ve integer phase clock (IPC) modellerini aynı gösterimle türetir. Kullanıcı çözümünde üçünün **eşdeğer** olduğunu gösterir: hepsi aynı IF kod/faz ve MW fonksiyonlarını kullanır, aynı WL ve N1 belirsizliklerini çözer, uydu başına üç düzeltme ister ve serbestlik derecesi n − 5'tir. Farkları yalnızca kesirli faz kısmını giderme yoludur.

| | SDBS (`PPP-AR/SDBS`) | DCM (`PPP-AR/DCM`) | IPC (`PPP-AR/IPC`) |
|---|---|---|---|
| Uydu düzeltmeleri (`code_m`, `phase_m`, `mw_m`) | Kod saati, SD N1 FCB, SD WL FCB | Kod saati, faz saati, MW biası | Kod saati, faz saati, WSB (DCM ürünü de kabul edilir) |
| Alıcı terimleri | Tek farkla yok olur | Kod, faz ve MW saati (3 durum) | Kod ve faz saati (2 durum); WRB uyduların ortalamasından |
| WL çözümü | Filtrede SD MW (makale: zaman ortalaması + yuvarlama, denk. 27) | Filtrede, MW alıcı biası ile birlikte | Filtre dışında: arc ortalaması, dairesel ortalama WRB, yuvarlama (denk. 46–47) |
| N1 çözümü | SD N1 filtrede | Farksız N1, referans uydu datumu | Farksız N1, WL bilinen kabul edilir (denk. 48) |
| Satır / bilinmeyen (ZWD dahil) | 3(n−1) / 4 + 2(n−1) | 3n / 7 + 2(n−1) | 2n / 6 + (n−1) |
| Serbestlik derecesi | n − 5 | n − 5 | n − 5 |

Kontroller ve değişiklikler (`processing/ar/solver.py`, `processing/ar/ipc.py`):

- **Fonksiyonel model makaleyle uyumlu:** `λ_N = c/(f1+f2) = 17 λ_IF` ve `β = c f2/(f1²−f2²) = 60 λ_IF`, yani `λ_IF N_IF = 17 λ_IF N1 + 60 λ_IF N_WL` (denk. 14). DCM'nin üç alıcı saati denk. 23–25 ile, SDBS'nin tek fark operatörü denk. 26–39 ile aynıdır.
- **Düzeltilen sorun, referans uydu değişimi:** Eski sürüm referans değiştiğinde tüm belirsizlikleri sıfırlıyordu. Makaleye göre datum farkı bir tamsayı kaymasıdır ve alıcı saatleri tarafından emilir. Artık yeni referans izleniyorsa durum `N_j ← N_j − N_ref` ile dönüştürülür (kovaryans da). Yakınsamış ve fix'lenmiş belirsizlikler korunur. Sentetik testte eski sürüm değişimden sonra en az `ar_min_epochs` epok boyunca float kalıyordu, yeni sürüm fix'i koruyor.
- **Yeni model, IPC:** WL, WSB ile düzeltilmiş MW'nin arc ortalamasından bulunur. Alıcı WL biası, kesirli kısımların dairesel ortalamasıdır (±0.5 civarında kesirlerin birbirini götürmemesi için). Arc yeterince uzun (`ar_min_epochs`), ortalamanın σ'sı 0.1 çevrimden küçük ve kalan kesir 0.25 çevrimden küçükse WL yuvarlanır ve referans uyduya göre sabit tutulur. N1 aşaması diğer modellerle aynı tamsayı kestiricilerini (LAMBDA, bootstrapping, cascade…) ve kabul testlerini kullanır.
- **Eşdeğerlik testi** (`tests/test_ar_models.py`): Aynı sentetik veride üç model doğru WL ve N1 tamsayılarını buluyor; son konumlar birbirinden 1 cm, gerçek konumdan 2 cm içinde. IPC'nin kestirdiği alıcı WL biası gerçek değere eşit.
- **Pratik farklar** (makalenin tartışma bölümü): SDBS'de alıcı saati tahmin edilmez. DCM WL'yi filtre içinde ve diğer bilinmeyenlerle birlikte kestirir. IPC'nin WL adımı basit ve hızlıdır, ama alıcı WL biasının her uydu için aynı olduğu varsayımına dayanır. Yanlış bir WL, IPC'de N1 aşamasına sabit olarak girer; bu yüzden WL kapıları tutucu tutulmuştur.

## Kod biası ve alıcı saat sıçraması (Banville et al. 2008)

Banville, Santerre, Cocard & Langley (2008) iki şeyi gösterir: (1) kod biasları hem MW geniş şeridine hem de IF uydu saatleri üzerinden IF belirsizliklerine girer (denk. 9–12), bu yüzden kullanıcının kodu ürünün üretildiği kodla aynı olmalıdır; (2) alıcı saat atlamasında bütün uyduların kesirli belirsizlikleri birlikte sıçrar. Yazılımda karşılıkları:

- **Kod biası hizalaması** (`gps_processing/io/bias_sinex.py`, `--code-biases`): Bias-SINEX 1.00 dosyasındaki uydu DSB/OSB değerleri okunur (ters yönlü DSB ve OSB çiftleri de kullanılır, alıcı satırları atlanır). C1W/C2W dışındaki kodlar (C1C, C1X, C2L, C2X…) `C − DSB(C, CW)` ile W eşdeğerine çevrilir; Pc, A4 ve yayım zamanı kodu doğrusal olduğu için doğrudan düzeltilir. Faz değiştirilmez: IGS kuralında bir frekansın uydu faz biası izleme modundan bağımsızdır, alıcı tarafındaki farklar tüm uydular için ortaktır. AR ürün kontrolü hizalanmış satırlarda kodun tam, fazın yalnız frekans olarak eşleşmesini ister. DSB'si olmayan satırlar hizalanmaz ve AR ürün kontrolünde reddedilir.
- **Saat sıçraması onarımı** (`gps_processing/processing/clock_jump.py`, varsayılan açık, `--no-clock-jump-repair`): Ardışık epoklarda kesintisiz uydular için Δ(Pc − Lc) hesaplanır. Medyan 1 ms'nin tam katına yuvarlanır; uyduların en az %80'i bu değere 30 m içinde uyuyorsa sıçrama kabul edilir ve tam `k·c·1 ms` değeri o epoktan sonra Lc ve A4'e eklenir. Kod ve zaman etiketi değişmez, GF etkilenmez. Bulunan sıçramalar `summary.json` → `settings.clock_jumps` alanına yazılır. Sentetik testte onarımsız DCM çözümü sıçramada bütün arc'ları yeniden başlatıp `ar_min_epochs − 1` epok float kalıyor; onarımlı çözüm fix'i koruyor.

## Mevcut sınırlar ve sonraki bilimsel adımlar

Troposfer kestirimi, ANTEX, phase wind-up, relativistik düzeltmeler, yüklemeler ve gerçek hassas saat/bias sağlayıcı adaptörleri bu değişiklikte tamamlanmadı. Yörünge yayım zamanı hâlâ SP3 saatiyle hesaplanır. Bu nedenle gerçek veride elde edilen herhangi bir `partial_fixed` etiketi jeodezik doğruluk kanıtı değildir. Uzun süreli gerçek gözlemler, bağımsız istasyon koordinatları ve uyumlu ürünlerle dış doğrulama gerekir. Ağ FCB kestirimi, IPC/OSB yöntemi ve FFRT için ayrı sonraki modüller planlanabilir; şu an seçenek olarak tamamlanmış gösterilmezler.
