# Faz 0, madde 0.5: olasılık metrikleri ve iki sayısal gerçekleşme

**Tarih:** 18 Eylül 2026
**Kaynak:** `results_v2/rerun_probs/T1_3class__N2` (T1, N2, 5x5 CV, 12 thread)
**Betik:** `src/phase0_probscores.py`
**Çıktılar:** `results_v2/phase0/probscores_{f1_macro,log_loss,brier}.csv`,
`results_v2/phase0/thread_realisation.csv`

Yapılandırma karşılaştırması: yeniden koşunun `config.json` dosyası orijinal koşununkiyle
23 alanın 23'ünde aynı. Değişen tek şey çalışma zamanı thread sayısı (orijinal 2, yeniden
koşu 12) ve olasılıkların kaydedilmesi.

---

## 1. Neden olasılık metrikleri

Makro F1, modelin kararını ikili hale getirir. Doğru sınıfa yüzde 51 olasılık veren bir
model ile yüzde 99 veren model aynı F1'i alır. Olasılık tabanlı metrikler bu bilgiyi korur,
dolayısıyla daha ince ayrım yapabilmeleri beklenir.

- **Log loss:** doğru sınıfa verilen olasılığın negatif logaritması. Emin ve yanlış olan
  tahmini çok sert cezalandırır, bu yüzden uç değerlere duyarlıdır.
- **Brier:** olasılık vektörü ile tek-sıcak hedef arasındaki kare hata. Sınırlı bir
  ölçekte kalır, daha kararlıdır.

Soru şuydu: F1 ile ayırt edilemeyen füzyon operatörleri, olasılık metrikleriyle ayrılıyor mu?

## 2. Sonuç: kısmen, ve beklenmedik yönde

| Metrik | Anlamlı çift (naif) | Anlamlı çift (düzeltilmiş) | Füzyon çiftlerinde anlamlı (10 çiftten) |
|---|---|---|---|
| makro F1 | 22 | 11 | 1 |
| log loss | 26 | 9 | 0 |
| Brier | 26 | 14 | 3 |

**Brier en güçlüsü, log loss en zayıfı.** Bu ters gibi görünüyor ama açıklaması var: early
füzyonun log loss dağılımı çok ağır kuyruklu (ortalama 0.406, standart sapma 0.192, yani
sapma ortalamanın yarısı kadar). Log loss emin yanlışları sert cezalandırdığı için birkaç
kötü fold tüm dağılımı savuruyor ve varyans şişiyor. Brier sınırlı olduğu için bu olmuyor.

**Yöntemsel çıkarım:** bu tür karşılaştırmalarda birincil olasılık metriği Brier olmalı,
log loss ikincil olarak raporlanmalı. Bu, ileride yazılacak metinde gerekçesiyle
belirtilmeli.

## 3. Çekirdek iddia ayakta

Brier'in ayırdığı 3 füzyon çiftinin **üçü de early füzyonu içeriyor**: early-gated
(fark 0.120), early-late (0.131), early-score (0.115). Ayrıca early-spec2d de anlamlı.

Yani:

- **early füzyon diğerlerinden ayrılıyor ve aşağı yönde ayrılıyor.** Bu bulgu F1'de zayıftı
  (yalnızca early-late anlamlıydı), Brier'de güçleniyor. Olasılık metrikleri early'nin
  yalnızca daha çok yanılmadığını, aynı zamanda yanılırken daha emin olduğunu gösteriyor.
- **late, gated, attention, score arasında hiçbir çift hiçbir metrikte ayrılmıyor.**
  Projenin merkezi iddiası üç metrik altında da ayakta.

Denklik sınırları hâlâ geniş: füzyon çiftlerinde düzeltilmiş δ_min, F1 için 0.022 ile 0.130
arasında. Yani "bu operatörler ±0.022 içinde denktir" diyebiliyoruz, "±0.01 içinde denktir"
diyemiyoruz. ROPE 0.01 için Bayesçi denklik olasılığı en fazla 0.61.

## 4. İkinci sayısal gerçekleşme: asıl bulgu

Aynı kod, aynı veri, aynı yapılandırma, aynı tohum. Tek fark: PyTorch'un kaç thread
kullandığı.

| Model | Ortalama F1 (2 thread) | Ortalama F1 (12 thread) | Ortalama fark | En büyük fold farkı | Birebir aynı fold |
|---|---|---|---|---|---|
| raw1d | 0.9511 | 0.9616 | +0.0105 | 0.0934 | 11/25 |
| spec2d_wide | 0.9655 | 0.9702 | +0.0047 | 0.0701 | 8/25 |
| early | 0.8963 | 0.8886 | -0.0077 | 0.0620 | 2/25 |
| late | 0.9740 | 0.9778 | +0.0038 | 0.0475 | 5/25 |
| score | 0.9637 | 0.9686 | +0.0049 | 0.0474 | 8/25 |
| gated | 0.9731 | 0.9728 | -0.0003 | 0.0449 | 8/25 |
| spec2d | 0.9602 | 0.9652 | +0.0049 | 0.0345 | 4/25 |
| attention | 0.9688 | 0.9694 | +0.0006 | 0.0323 | 6/25 |
| raw1d_wide | 0.9628 | 0.9647 | +0.0019 | 0.0207 | 11/25 |
| logvar | 0.6460 | 0.6460 | 0.0000 | 0.0000 | 25/25 |
| shallow | 0.9536 | 0.9536 | 0.0000 | 0.0000 | 25/25 |

Sığ referanslar (logvar, shallow) PyTorch kullanmadığı için birebir aynı. Bu, farkın
kaynağının kayan nokta toplama sırası olduğunu doğruluyor, kodda bir tohumlama hatası
olmadığını gösteriyor.

### 4.1 Ölçek karşılaştırması

Çekirdek füzyon operatörleri arasındaki tüm yayılım (late 0.9778, score 0.9686 arası)
**0.0092**.

raw1d modelinin yalnızca thread sayısı değiştiği için kayması **0.0105**.

**Yani ilgisiz bir çalışma zamanı ayarının tek bir modelde yarattığı kayma, karşılaştırılan
operatörlerin tamamının birbirinden ayrıldığı aralıktan daha büyük.**

### 4.2 Yön değiştiren karşılaştırmalar

55 model çiftinden **3'ü işaret değiştirdi**, yani iki gerçekleşmede zıt sonuç veriyor:

| Çift | 2 thread | 12 thread | Anlamı |
|---|---|---|---|
| attention vs spec2d_wide | +0.0033 | -0.0008 | Füzyon, kapasite eşlenmiş tek modaliteli modeli bir koşuda yeniyor, diğerinde yeniliyor |
| raw1d vs shallow | -0.0025 | +0.0080 | Derin 1B CNN, 7 öznitelikli lojistik regresyonu bir koşuda yeniyor, diğerinde yeniliyor |
| raw1d_wide vs spec2d | +0.0026 | -0.0005 | Hangi modalitenin daha iyi olduğu değişiyor |

Birinci satır özellikle önemli: **"füzyon kapasite eşlenmiş tek modaliteli modelden iyi
midir"** sorusu, tam olarak literatürün sorduğu soru. Cevabı bu deneyde thread sayısına
bağlı.

İkinci satır da çarpıcı: derin ağın basit bir sığ modeli yenip yenmediği de öyle.

Çekirdek operatör sıralaması (late > gated > attention > score) iki gerçekleşmede de aynı
kaldı. Yani kaos değil, ama küçük farklara dayanan her sonuç kırılgan.

## 5. İddialara etkisi

| İddia | Durum |
|---|---|
| late, gated, attention, score ayırt edilemiyor | **Güçlendi.** Üç metrik altında da ayrılmıyorlar |
| early füzyon geride kalıyor | **Güçlendi.** Brier'de 4 modelden anlamlı biçimde ayrılıyor |
| Füzyon, kapasite eşlenmiş tek modaliteli modelden iyi değil | **Kırılgan.** Fark işareti gerçekleşmeye bağlı, bu da zaten "ayırt edilemiyor" demenin başka bir yolu |
| logvar en kötü | **Ayakta.** Her metrikte, her gerçekleşmede |
| Belirli sayılar (0.9740 gibi) | **Tek başına raporlanamaz.** Üçüncü ondalık basamak thread sayısına bağlı |

## 6. Yapılması gerekenler

1. **Thread sayısı yapılandırmaya yazılmalı** ve hash'e dahil edilmeli. Şu anda
   `config.json` bunu kaydetmiyor, bu bir tekrarlanabilirlik açığı.
2. **Sonuçlar tek ondalık hassasiyetle fazla raporlanmamalı.** Üçüncü basamak gürültü.
3. **Birincil olasılık metriği Brier** olmalı, log loss ikincil.
4. **Bu bulgu makalenin katkılarından biri olmalı.** "Değerlendirme pratiği bu büyüklükteki
   farkları ayırt edemiyor" tezinin en somut kanıtı bu: farkı üretmek için modeli bile
   değiştirmek gerekmiyor, thread sayısını değiştirmek yetiyor.
5. **Temiz yeniden kurulumda** thread sayısı sabitlenmeli ve kaydedilmeli; ayrıca en az iki
   farklı thread sayısıyla koşulup fark raporlanmalı.

## 7. Ek bulgu: duyarlılık koşuları farklı thread sayısında çalışmış

Koşu betikleri incelendiğinde ortaya çıktı:

- `run_all.sh` (ana koşular): `THREADS=${THREADS:-2}`
- `run_phase2.sh` (duyarlılık koşuları): `THREADS=${THREADS:-3}`

Yani `results_v2/sensitivity` altındaki dört duyarlılık noktası, ana koşulardan farklı bir
sayısal ortamda üretilmiş. Bu, duyarlılık analizinin sonucunu geçersiz kılmaz, çünkü orada
sorulan soru "aynı tasarım noktasında operatörler ayrılıyor mu" idi ve bu karşılaştırma
kendi içinde tutarlı. Ancak **duyarlılık noktalarındaki mutlak sayılar ana koşunun
sayılarıyla üçüncü ondalık basamakta karşılaştırılamaz.**

Düzeltme: iki betik de 2'ye sabitlendi ve thread sayısı artık koşu hash'ine dahil, yani
bundan sonra farklı thread sayısıyla üretilen sonuçlar ayrı dizinlere yazılıyor ve
birbirinin üzerine yazamıyor.

## 8. Yapılan kod değişiklikleri (18 Eylül 2026)

| Değişiklik | Dosya | Gerekçe |
|---|---|---|
| `threads` alanı Config'e eklendi, hash'e dahil | `src/config.py` | Sonucu değiştiren bir ayarın koşu kimliğinde olmaması, sessiz üzerine yazma riski taşıyordu |
| `runtime_env()` eklendi, her koşu `env.json` yazıyor | `src/config.py` | torch/numpy/sklearn sürümü, platform, çekirdek sayısı, fiilen yürürlükteki thread sayısı |
| `outdir()` artık dizin açmıyor | `src/config.py` | İsim hesaplamanın yan etkisi boş artık dizinler bırakıyordu |
| `run()` thread sayısını config'ten set ediyor | `src/run_benchmark.py` | Çağıran kim olursa olsun aynı değer yürürlükte olsun |
| `phase0.load()` birden fazla eşleşmede uyarıyor | `src/phase0.py` | Hash değiştiği için aynı kalıba birden fazla dizin uyabilir; sessizce yanlışını seçmesin |

**Not:** Config'e alan eklendiği için hash değişti. `T1_3class__N2_z_zspec` için eski hash
`a45d0ab077`, yenisi `5ef438ef5b`. Eski sonuç dizinleri olduğu gibi duruyor ve analiz
betikleri onları bulmaya devam ediyor, çünkü arama görev ve normalizasyon ön ekiyle
yapılıyor. `python -m src.phase0` çalıştırıldı, sonuçlar değişmedi.
