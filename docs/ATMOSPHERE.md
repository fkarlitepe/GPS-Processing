# External atmosphere products

Two optional files can be selected in **Processing settings → External atmosphere products**
or passed with `--ionosphere` / `--troposphere`. Both must be inside `Data/`; `.gz` is read
directly. They apply to the EKF estimators (float PPP and both PPP-AR models). The
`Sequential LS (v2.0)` baseline stays unchanged and ignores them.

## Troposphere — SINEX_TRO (`*.TRO`, `*.zpd`)

| Step | Algorithm |
|---|---|
| Station | RINEX `MARKER NAME` (first 4 characters), otherwise the nearest station in `+TROP/STA_COORDINATES` within 20 km |
| Columns | `TROTOT` (ZTD), its `STDDEV`, optional `TGNTOT` / `TGETOT`; units from `TROPO PARAMETER UNITS` (IGS: mm) |
| Time | linear interpolation of ZTD and gradients; epochs more than 1 h outside the file are rejected |
| Split | ZHD from Saastamoinen (standard atmosphere), a-priori ZWD₀ = ZTD − ZHD |
| Slant | ZHD·m_h(e) + ZWD₀·m_w(e) (Niell varsayılan, Chao seçenek) + m_g(e)·(G_N cos a + G_E sin a), m_g = 1/(sin e tan e + 0.0032) (Chen & Herring 1997) |
| Filter | ZWD estimation on: the state is the residual δZWD, starting at 0 with σ = 2·max(product σ, 5 mm). Off: the product delay is applied as fixed |

The reported `zwd_m` is always the total wet delay (ZWD₀ + δZWD).

## Ionosphere — IONEX (`*.INX`, legacy `*.yyI`)

| Step | Algorithm |
|---|---|
| Maps | TEC maps only, `EXPONENT` applied, 9999 → missing |
| Pierce point | single-layer model on the IONEX shell (`HGT1`, `BASE RADIUS`), spherical geometry |
| Slant factor | F = 1/√(1 − (R cos e /(R+H))²) (`COSZ`) |
| VTEC | bilinear in latitude/longitude, linear in time between the two nearest maps rotated with the Sun (λ + Δt·360°/day; IONEX 1.0 recommendation) |
| Use in PPP | the ionosphere-free combination already removes the first-order term (40.3·STEC/f²), so the map provides the **second-order** term |
| Second order | s = 7527·c·(B·k)·STEC with B from a centred dipole (IGRF-13 2020: g10, g11, h11) at the pierce point and k the satellite→receiver direction (IERS Conventions 2010 §9.4; Hernández-Pajares et al. 2007) |
| IF delays | code −s/(f₁f₂(f₁+f₂)), phase +s/(2f₁f₂(f₁+f₂)); typical size: a few mm, up to ~2 cm at high TEC |

The Melbourne–Wübbena row of the PPP-AR models is left uncorrected because its residual
second-order term is far below the code noise.

New position columns: `stec_tecu` (mean slant TEC of the epoch) and `iono2_phase_mm`
(mean |second-order IF phase delay|). `summary.json → settings` records the models used.

## Not included

Third-order ionosphere and ray bending (< 1 mm), the full IGRF field instead of the
dipole, VMF1/VMF3 or GPT grids, and the use of the GIM as a constraint in an uncombined
(L1/L2) or single-frequency model. The present filters are ionosphere-free only.

## Where to get the files

- GIM: IGS final `IGS0OPSFIN_YYYYDDD0000_01D_02H_GIM.INX.gz`, or CODE `COD0OPSFIN_..._01H_GIM.INX.gz`
- Troposphere: IGS final station product, `IGS0OPSFIN_YYYYDDD0000_01D_05M_<SITE>_TRO.TRO.gz`

These are available from the IGS data centres (CDDIS, IGN, BKG). Pick the files for the
observation day.

## Troposfer eşleme fonksiyonu (Niell 1996)

Shi & Gao (2012) ile aynı yapı kullanılır: Saastamoinen ZHD önsel değerdir, Niell eşlemesiyle eğik doğrultuya taşınır ve modelden kalan ZWD filtrede rastgele yürüyüş olarak kestirilir. Troposfer yalnız modelle sabitlenmez. Standart atmosferli modelin cm düzeyindeki ıslak gecikme hatası, düşük yükseklik açısında eşleme ile 5–10 kat büyür ve N1 dalga boyunu (≈10.7 cm) aşar.

- **Niell (varsayılan, `--mapping niell`):** Hidrostatik ve ıslak katsayılar enleme göre (15°–75°) doğrusal enterpolasyonla bulunur. Hidrostatik kısma yıllık terim (faz DOY 28, güney yarımkürede +182.625 gün) ve yükseklik düzeltmesi eklenir. Gün bilgisi gözlem tarihinden alınır.
- **Chao (`--mapping chao`):** Yalnız yükseklik açısına bağlıdır; v2.x sonuçlarıyla karşılaştırma için tutuldu.
- **Karşılaştırma:** İki fonksiyon 15° üstünde ‰ düzeyinde, 5°'de yaklaşık %0.5 (kuru) ve %3 (ıslak) farklıdır. 13° maskeli 12 dakikalık MADR verisinde faz artıkları aynı (8.41 mm), ZWD farkı 6 mm.
- **İyonosfer:** IF kombinasyonu birinci derece etkiyi yok ettiği için Klobuchar gibi bir model IF çözümünde etkisizdir. İkinci derece terim IONEX ile düzeltilir.
