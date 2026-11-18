# Uydu seçimi ve DOP / ağırlıklı DOP analizi (v3.0, float PPP)

Float PPP'de (EKF) her epokta görünen uydulardan yalnız **k** tanesinin filtre güncellemesine girmesi seçilebilir. Seçim yapılmasa bile her epok için DOP/WDOP analizi yazılabilir. Kod: `gps_processing/processing/selection.py`. Motora bağlantı: `processing/engine.py`. Rapor: `analysis.selection_report`.

## Akış

1. Yükseklik maskesi, kayma (cycle slip) tespiti ve yay sayacı **bütün** görünen uydularla çalışır. Böylece kullanılmayan bir uydunun kayması da yakalanır.
2. `SatelliteSelector.select` alt kümeyi belirler. Yalnız bu uydular kod ve faz satırı olarak güncellemeye girer.
3. Kullanılmayan uyduların belirsizlik durumları korunur: gözlenmez, süreç gürültüsü yoktur. Uydu tekrar seçildiğinde kaldığı yerden devam eder.
4. Güncellemeden sonra `record_update`, kullanılan her uydunun model terimlerini, artıklarını ve Kalman düzeltmesindeki payını kaydeder.

## Geometri ve DOP

Satır: g_i = [ −e_i(ENU) | sistemin alıcı saati (GPS, Galileo) | m_w(el_i) ]. Son sütun yalnız ZWD kestiriliyorsa vardır; `--no-selection-troposphere` ile çıkarılabilir.

Q = (Gᵀ W G)⁻¹. Seçimde bulunmayan bir sistemin saat sütunu ayrıştırılır (birim köşegen).

| Ölçüt | Tanım |
|---|---|
| GDOP | √iz Q (konum, saatler, ZWD) |
| PDOP | √(Q_EE + Q_NN + Q_UU) |
| HDOP | √(Q_EE + Q_NN) |
| VDOP | √Q_UU |

**Ağırlıklar** (`--selection-weighting`). W = I iken klasik DOP elde edilir; diğer seçenekler ağırlıklı DOP'tur (WDOP).

| Seçenek | w_i | Anlamı |
|---|---|---|
| `none` | 1 | Klasik DOP |
| `elevation` | sin²(el) | Filtrenin yükseklik modeli |
| `cn0` | 10^((C/N0 − 45)/10) | σ² = a·10^(−C/N0/10); w = 1, 45 dB-Hz'de |
| `filter` | (σ_ref / σ_i)² | Filtrenin kendi kod varyansı: IF çarpanı ve, açıksa, ML stokastik modeli dahil. σ_ref = tepe noktasındaki GPS uydusu, böylece WDOP ile DOP aynı ölçekte kalır |

C/N0 için açık servis L1/E1 sinyali kullanılır: GPS S1C, Galileo S1C/S1X (`cn0` sütunu). Bazı alıcıların yarı-kodsuz S1W değerleri sistemler arasında karşılaştırılabilir değildir.

## Yöntemler (`--selection`)

| Yöntem | Açıklama |
|---|---|
| `all` | Seçim yok (yalnız analiz) |
| `fibo-gnss` | Altın açılı Fibonacci küresi. (1) max(2k, 12) hücrede en iyi kaliteli uydular (hücredeki en iyi ağırlığın %80'i) havuza alınır. (2) N ∈ {k, k+1} referans yön, 2 spiral yönü ve 12 döndürme ile 48 nokta kümesi üretilir; her küme, açı − 0,15·ln(w/w_max) maliyetine göre açgözlü biçimde uydulara eşlenir. (3) Tek uydulu sistemler atılır, alt küme açgözlü ekleme ile k'ya tamamlanır. (4) En küçük ölçütlü aday seçilir |
| `fibo-gnss+ls`, `backward+ls` | En iyi iyileştirmeli 1-takas yerel arama (en fazla 20 adım) |
| `backward` | Açgözlü geri eleme: her adımda çıkarılınca ölçütü en az artıran uydu atılır |
| `top-cn0` | En yüksek C/N0'lı k uydu |
| `az-sector` | En yüksek uydu + k−1 azimut diliminin her birinden en alçak uydu |
| `random` | Rastgele alt küme; alt sınır olarak kullanılır |
| `exhaustive` | Tam tarama. 300 000 alt kümeden fazlası varsa fibo-gnss+ls'ye düşer |

Her yöntem aynı uygulanabilirlik kurallarına uyar: bir sistem ya hiç yoktur ya da en az 2 uydu ile vardır, ve konum kestirilebilir olmalıdır.

**Faz yaylarını korumak.** Belirsizlikler yalnız uydu kullanılırken yakınsar. Her epok farklı bir alt küme seçmek PPP'nin yakınsamasını bozabilir. Bunu sınırlayan iki seçenek var:
- `--selection-hysteresis 0.05`: Yeni alt küme ölçütü en az %5 düşürmüyorsa eski alt küme tutulur.
- `--selection-interval 600`: Alt küme 600 s boyunca, uyduları görünür kaldıkça tutulur.

Seçilen uydulardan biri batarsa alt küme yeniden seçilir.

## Analiz çıktıları

| Dosya | İçerik |
|---|---|
| `selection_epochs.csv` | Epok başına: görünen ve kullanılan uydu sayıları (sistem başına), ölçüt; tüm görünen ve seçilen uydular için GDOP/PDOP/HDOP/VDOP ve W-karşılıkları; yeniden seçim; süre (ms) |
| `selection_satellites.csv` | Her epok ve görünen uydu: yükseklik, azimut, C/N0, ağırlık, `selected` ve `delta_dop`. Kullanılan uydular için ek olarak: kod/faz artıkları, troposfer eğik gecikmesi, uydu saati, rölativite, 2. derece iyonosfer, belirsizlik ve Kalman payı (`correction_e/n/u_m`) |
| `selection_per_satellite.csv` | Oturum boyunca uydu başına: kullanılma oranı, ortalama yükseklik, C/N0 ve ağırlık, medyan Kalman payı, artık RMS'leri, medyan `delta_dop` |
| `selection_report.md` | Özet tablo ve uydu tablosu |
| `selection_dop.svg`, `selection_counts.svg`, `selection_skyplot.svg` | DOP/WDOP zaman serisi, uydu sayıları, gökyüzü grafiği (seçilen ve dışlanan uydular) |

`delta_dop` için iki durum vardır:
- Kullanılan bir uydu için: uydu çıkarılırsa ölçütün ne kadar arttığı (geometrik değeri).
- Dışlanan bir uydu için: uydu eklenseydi ölçütün nasıl değişeceği (negatif değer: eklenmesi yardımcı olurdu).

**Kalman payı.** Konum güncellemesi Δx = K·v'dir. Uydu i'nin payı K_pos,i·ṽ_i olarak tanımlanır; burada ṽ_i = v_i − h_i,saat·Δx_saat, yani ön-uyuşumsuzluktan saat güncellemesi çıkarılmıştır.
- Alıcı saati her epok beyaz gürültü olarak sıfırlandığı için ham v_i ortak saat sapmasını (onlarca km) taşır. Bu sapma toplamda birbirini götürür, ama tek tek paylara girer.
- Saat kısmı ayrılınca payların toplamı gerçek konum güncellemesine eşittir. MADR'de doğrulandı: fark 0,0 mm, korelasyon 1,0.
- İlk epoklarda paylar büyüktür (başlangıç yakınsaması), bu yüzden özetler medyan kullanır.

Arayüzde ayarlar *Processing settings → Satellite selection and DOP / weighted-DOP analysis* bölümündedir. Sonuçlar *Results → Satellite selection / DOP* sekmesinde görünür: DOP/WDOP grafikleri, uydu tablosu ve seçilen bir epokta görünen ve kullanılan uydular. Karşılaştırma sekmesinde `Selection · …` varyantları vardır; k ayarlanmamışsa 8 kullanılır.

## Komut satırı

```bash
python -m gps_processing --sp3-day Data/products/2023327/COD/COD0MGXFIN_20233270000_01D_05M_ORB.SP3 \
  --clock-file Data/products/2023327/COD/COD0MGXFIN_20233270000_01D_30S_CLK.CLK --antex Data/models/igs20_2287.atx \
  --rinex Data/obs/MADR00ESP_R_20233270000_01D_30S_MO.rnx --systems G E \
  --selection fibo-gnss+ls --selection-count 8 --selection-weighting filter --selection-hysteresis 0.05
```

Seçim olmadan yalnız analiz için `--selection-analysis` kullanılır.

## Deney sonuçları (Deney S, `experiments/run_selection.py`)

8 EUREF/IGS istasyonu (AJAC, BRUX, EBRE, ENAO, GRAZ, MADR, SOFI, WTZR), 23.11.2023, 2 saatlik oturumlar (istasyon başına 12, toplam 96), GPS + Galileo, COD MGEX final ürünleri, statik float PPP (EKF). Yakınsama ölçütü: yatay ve düşey hata ≤ 10 cm, 20 epok boyunca. Doğruluk ölçütü: oturumun son 30 dakikasının RMS'i. Değerler oturumların medyanıdır; ayrıntılı tablo `results/experiments/selection/summary.md` dosyasındadır.

**Tüm uydular, alt kümeden daha iyi.** Bütün görünen uydular (ortalama ≈ 15) kullanıldığında yakınsama 14,5 dk, 3B RMS 18,3 mm'dir. Tablodaki satırlar FIBO-GNSS+LS ve filtre WDOP ile k uydu kullanıldığındaki sonuçtur; parantez içi fark tüm uydulara göredir:

| k | Yakınsama (dk) | 3B RMS (mm) |
|---|---|---|
| 6 | 33,5 (+19,5) | 31,5 (+15,9) |
| 8 | 27,8 (+12,0) | 25,9 (+6,7) |
| 10 | 22,5 (+7,0) | 22,7 (+2,9) |

Bütün farklar anlamlıdır (p < 0,001). PPP'de uydu azaltmak, belirsizliklerin ayrışması için gereken artıklığı (redundancy) azaltır.

**WDOP'un etkisi.** Aynı yöntem ve k ile ağırlıksız (klasik) DOP seçimine karşı, aynı istasyon ve oturum üzerinden eşleştirilmiş Wilcoxon testi:

| k | Etki |
|---|---|
| 6 | 3B RMS 1–8 mm (%3–17) düşer; 6 sonucun 5'i anlamlı (yalnız FIBO-GNSS+LS ile filtre ağırlığı değil). Yakınsama 2–3 dk kısalır |
| 8 | 3B RMS 3–7 mm (%10–20) düşer; 6 sonucun 6'sı anlamlı (p ≤ 0,03). Yakınsamada fark çoğunlukla anlamlı değil |
| 10 | Etki 0,3–2 mm'ye iner ve anlamlı değil |

Ağırlıklar arasında belirgin fark yoktur. C/N0 ve sin²el en az filtre ağırlığı kadar iyidir. Klasik DOP, bu ağırlıklardan hiçbirini kullanmadığı için alçak ve gürültülü uyduları seçer: alt kümenin PDOP'u en küçüktür (k = 8'de 3,5'e karşı 4,0), ama PPP sonucu daha kötüdür.

**DOP, PPP sonucunu öngörmüyor.** Oturum medyan PDOP/WPDOP değeri ile yakınsama, 3B RMS veya σU arasındaki Spearman korelasyonu |ρ| ≤ 0,17'dir. Benzer biçimde, 600 s tutulan RANDOM alt küme 3B RMS'te ağırlıksız DOP seçiminden daha iyidir (k = 8: 25,4'e karşı 32,8 mm). Az sayıda faz yayı kesintisi ve ölçüm kalitesi, anlık geometriden daha belirleyicidir.

**FIBO-GNSS ve BACKWARD.** Yalnız k = 6'da ve ağırlıksız DOP ile FIBO-GNSS+LS daha iyidir: −7,8 mm ve −1,5 dk (p ≤ 0,03). k = 8 ve 10'da iki yöntem neredeyse aynı alt kümeyi bulur ve sonuçları aynıdır. FIBO-GNSS seçimi bir epokta yaklaşık 20 ms, BACKWARD+LS yaklaşık 10 ms sürer.

**Histerezis.** k = 8, FIBO-GNSS+LS ve filtre ağırlığıyla %5 histerezis yeniden seçimleri yarıya indirir (20'ye karşı 43). Ama yakınsama 0,5 dk daha uzundur (p = 0,005) ve 3B RMS'te anlamlı fark yoktur. Bu nedenle %5'lik eşik yararlı değildir; yayları korumak için daha uzun bir `--selection-interval` denenmelidir.

**Sonuç.** Float PPP'de uydu seçimi yalnız işlem yükü veya bant genişliği sınırı varsa anlamlıdır. Seçim yapılacaksa ağırlıklı DOP (WDOP) kullanılmalıdır: k ≤ 8'de 3B hatayı %10–20 düşürür. k ≥ 10'da ağırlıklandırmanın önemi kalmaz.
