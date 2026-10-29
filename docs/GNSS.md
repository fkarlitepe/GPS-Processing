# GPS + Galileo (v2.8)

Uydu sistemleri çözümden önce seçilir: arayüzde *Positioning technique → Satellite systems* (GPS, Galileo onay kutuları), komut satırında `--systems G E`. Varsayılan yalnız GPS'tir.

## Sinyaller ve kimlikler

| Sistem | Frekanslar | IF, MW | Uydu kimliği (`gps_processing/gnss.py`) |
|---|---|---|---|
| GPS | L1 1575.42 / L2 1227.60 MHz | λ_W = 86.2 cm, λ_N = 10.7 cm | PRN n → n |
| Galileo | E1 1575.42 / E5a 1176.45 MHz | λ_W = 75.1 cm, λ_N = 10.9 cm | PRN n → 100 + n |

- **RINEX 3:** G ve E satırları ayrı gözlem tipleriyle okunur. Galileo'da önce X (data+pilot), sonra C/Q izleme modu seçilir.
- **RINEX 2.11:** İzleme modu yoktur; Galileo C1/L1 → C1X/L1X, C5/L5 → C5X/L5X kabul edilir. Alıcı aslında C1C veya C5Q izliyorsa kod biasları biraz farklıdır (aşağıdaki gerçek veri notuna bakın).
- Çıktı tablolarında tamsayı kimliğin yanında okunur ad (`prn`: G05, E12) bulunur.

## Modeller

- **Float PPP (EKF):** GPS saati + ayrı Galileo alıcı saati (beyaz gürültü; GPS–Galileo zaman farkı ve sistemler arası kod biası buna girer). Belirsizlik durumları 68 yuva (G01–G32, E01–E36). Gürültü, IF katsayısı ve ikinci derece iyonosfer sistemin frekanslarıyla hesaplanır.
- **PPP-AR:** Her sistemin kendi referans uydusu (datum) vardır. SDBS sistem içinde fark alır; DCM'de üç, IPC'de iki alıcı saati/biası sistem başına ayrıdır; IPC WL çözücüsü sistem başına ayrı alıcı WL biası kullanır. WL ve N1 adayları sistem içi fark olduğundan iki sistemin tamsayıları birlikte aranır. Referans değişiminde tamsayı kaydırma yalnız o sistemin belirsizliklerine uygulanır.
- **Anten ve wind-up:** ANTEX'te Galileo uyduları E01/E05 frekanslarıyla okunur. Alıcı kalibrasyonunda Galileo frekansı yoksa IGS alışkanlığıyla GPS değerleri kullanılır (E1 → G01, E5a → G02). Wind-up IF fazda sistemin λ_N değeriyle uygulanır.
- **Kod biası hizalaması (Bias-SINEX):** Yalnız GPS (C1C/C2L… → C1W/C2W). Galileo kodları olduğu gibi kalır; saat ürünleri merkezden merkeze C1X/C5X veya C1C/C5Q kullanır.
- **Sıralı LS (v2.0):** Yalnız GPS; Galileo seçiliyken bu kestirici reddedilir, karşılaştırmada temel çözüm GPS ile çalışır.

## Girdi kontrolleri

- Galileo seçiliyse SP3 ve CLK dosyasında Galileo kaydı olmalı (IGS final yörüngeleri yalnız GPS'tir; `*MGX*` gibi çok sistemli ürün gerekir). Yoksa çözüm açık bir mesajla durur.
- PPP-AR'de seçilen her sistem AR ürününde bulunmalı (OSB'de E1/E5a biasları, FCB'de Galileo satırları). Yoksa "ürün X'te Galileo biası yok; Galileo'yu kaldırın veya Galileo içeren ürün seçin" hatası verilir.
- FCB kestirimi her sistem için ayrı yapılır (alıcı UFO'ları sistemden sisteme farklıdır) ve sistem başına en az 5 istasyon ister. 5'ten az istasyonu olan sistem atlanır ve `summary.json → systems` altında bildirilir; hiçbir sistem yeterli değilse hata verilir. FCB dosyasında GPS ikinci frekansı L2, Galileo E5a L5 sütunundadır.
