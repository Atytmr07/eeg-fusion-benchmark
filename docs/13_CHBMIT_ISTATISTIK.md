# CHB-MIT LOSO: Düzeltilmiş İstatistik

**Tarih:** 21-22 Eylül 2026
**Kaynak:** `results_v2/chbmit/loso_main/perfold.csv` (24 denek, LOSO, **11 model**:
8 derin model `src/chbmit_run.py` ile, 2 klasik referans `src/chbmit_baselines.py`
ile (sklearn, 54 saniye), `score` operatörü `src/chbmit_score.py` ile (raw1d ve
spec2d'nin yeniden eğitilip doğrulama kümesinde seçilen ağırlıkla harmanlanması,
~4 saat). Artık Bonn ile birebir aynı model seti.
**Betik:** `src/chbmit_stats.py`
**Çıktı:** `results_v2/chbmit/phase0/chbmit_{f1_macro,log_loss,brier}.csv`

Test/eğitim oranı: LOSO'da 24 katman, her biri farklı bir denek dışarıda bırakılarak
üretiliyor. Oran 1/(n_katman-1) = 1/23. Bu, Bonn'daki k-katlı formülün doğrudan
uzantısı; düzeltmenin kaynağı katmanların eğitim verisini paylaşması, "k-fold" biçimine
özgü değil.

---

## 1. Özet: Bonn'dan daha güçlü bir "ayırt edilemiyor" sonucu

| | Bonn (T1, 5x5 CV, n=25) | CHB-MIT (LOSO, n=24) |
|---|---|---|
| Naif Wilcoxon, anlamlı çift (tüm modeller) | 22/55 | **0/55** |
| Düzeltilmiş test, anlamlı çift (5 füzyon operatörü, 10 çift) | 1/10 | **0/10** |
| Tüm modeller yayılımı (F1) | 0.332 | 0.068 |
| Füzyon operatörleri yayılımı (F1, 5 operatör) | 0.089 | 0.020 |

`score` eklendiğinden beri karşılaştırma tam olarak Bonn'la aynı beş operatörü
kapsıyor (early, late, gated, attention, score). CHB-MIT'te bu 10 çiftin **hiçbiri**
düzeltilmiş testte anlamlı çıkmıyor; Bonn'da 10 çiftin 1'i anlamlıydı (early'nin
diğerlerinden ayrılması).

**En çarpıcı fark:** Bonn'da naif test 22 "anlamlı" fark buluyordu, bunların çoğu Faz
0'da düzeltmeyle çöktü. CHB-MIT'te **naif test bile hiçbir fark bulmuyor.** Yani
CHB-MIT'teki "ayırt edilemiyor" sonucu, düzeltilmiş testin muhafazakârlığından
kaynaklanmıyor; deneklerarası değişkenlik o kadar büyük ki hiçbir istatistiksel yöntem
bu 8 modeli birbirinden ayıramıyor.

## 2. Ama bir nüans var: denklik de kanıtlanamıyor

Bu, "fark yok" ile aynı şey değil. Bonn'da denklik testi pozitif kanıt veriyordu (P(ROPE
0.01 içinde denk) bazı çiftlerde 0.61'e çıkıyordu). CHB-MIT'te bu olasılıklar **düşük**:

| Metrik | P(ROPE alt sınırı içinde denk), 5 operatör / 10 çift aralığı |
|---|---|
| f1_macro (ROPE 0.01) | 0.18 - 0.26 |
| log_loss (ROPE 0.02) | 0.00 - 0.07 |
| brier (ROPE 0.01) | 0.02 - 0.21 |

Yani CHB-MIT'te operatörler arasında **ne fark var diyebiliyoruz ne de denk diyebiliyoruz.**
Sonuç gerçekten belirsiz, kararsız. Sebep açık: 24 denek arası varyans çok büyük
(f1_macro standart sapması model başına 0.13-0.18, ortalama farkların kendisinden
kat kat büyük). Düzeltilmiş denklik sınırı da bunu yansıtıyor: f1_macro için δ_min
0.055-0.082, Bonn'daki 0.022'nin altına inemiyor.

**Bu, makalede ayrı ayrı raporlanması gereken iki farklı bulgu:** Bonn'da operatörler
istatistiksel olarak birbirine denk (pozitif kanıt var). CHB-MIT'te operatörler
istatistiksel olarak ayırt edilemiyor ama denklik de gösterilemiyor (kanıt yetersiz,
negatif sonuç değil belirsiz sonuç).

## 3. Beklenmedik bulgu: "early kötü" CHB-MIT'te tekrarlamıyor

Bonn'da early füzyon açık ara en kötüsüydü (F1 0.889, diğerleri 0.96-0.98). CHB-MIT'te
durum farklı:

| Model | Bonn F1 | CHB-MIT F1 |
|---|---|---|
| spec2d (tek modaliteli kontrol) | 0.965 | **0.695 (en iyisi)** |
| gated | 0.973 | 0.689 |
| score | 0.969 | 0.677 |
| late | 0.978 | 0.675 |
| attention | 0.969 | 0.672 |
| early | **0.889 (en kötüsü)** | 0.669 (orta sıra) |
| raw1d | 0.962 | 0.651 |
| raw1d_wide | 0.965 | 0.630 (en kötüsü) |

CHB-MIT'te en kötü performans **raw1d_wide**'da (kapasite eşlenmiş tek modaliteli
kontrol), early ortada bir yerde, beş füzyon operatörünün en kötüsü bile değil. Bonn'un
tutarlı bulgusu ("early diğerlerinden ayrı düşüyor") CHB-MIT'te tekrarlamıyor. Bu iki
türlü yorumlanabilir: (a) early füzyonun zayıflığı Bonn'a özgü bir yapıntı, (b)
CHB-MIT'teki gürültü seviyesi gerçek farkı gizliyor. Elimizdeki veriyle ikisini
ayıramıyoruz; bu da makalede dürüstçe yazılmalı.

## 4. log_loss yine kırılgan, ikinci kez doğrulandı

CHB-MIT'te log_loss standart sapması bazı modellerde 0.42-0.80 arası, yani
**ortalamanın kendisinden büyük.** Bu, Faz 0'da Bonn'da bulunan "log_loss ağır kuyruklu,
birincil metrik olmamalı" tespitini bağımsız bir veri setinde doğruluyor. Birincil
metrik olarak Brier kullanma kararı bu ikinci doğrulamayla güçlendi.

Küçük ama tutarlı bir gösterge daha: Brier'de naif Wilcoxon 3 çift buluyor, log_loss'ta
5 çift; düzeltilmiş testte ikisinde de hiçbiri kalmıyor. Bonn'daki gibi burada da naif
test optimist, hem Brier'de hem log_loss'ta.

**Beklenmedik bir sonuç:** `score` (raw1d ve spec2d'nin doğrulama kümesinde seçilen
ağırlıkla harmanlanması) log_loss'ta **11 modelin en iyisi** (0.501, ikincisi logvar
0.609) ve Brier'de gated ile neredeyse eşit en iyisi (0.2597'ye karşı 0.2596). Yani en
basit füzyon biçimi, hiçbir ek eğitim gerektirmeden, en iyi kalibre edilmiş model
çıkıyor. Bu ilginç ama dikkatli okunmalı: `score`'un F1'i (0.677) diğerlerinden
istatistiksel olarak ayrılmıyor, yalnızca kalibrasyonda öne çıkıyor. Karar sınırının
doğruluğu ile olasılık tahmininin güvenilirliği farklı şeyler; makalede bu ayrım
netleştirilmeli.

## 5. İkinci doygunluk bulgusu: klasik öznitelikler burada da rekabetçi

Bonn'un en çarpıcı bulgusu, tek öznitelikli bir modelin klasik görevde 0.954 F1
alıp en iyi derin modele (0.986) çok yaklaşmasıydı. CHB-MIT'te aynı testi
`src/chbmit_baselines.py` ile yaptık (126 boyutlu klasik öznitelik + lojistik
regresyon, kanal başına log-varyans/line-length/bant gücü, sklearn, 54 saniye):

| Model | F1 | Sıra (11 içinde) |
|---|---|---|
| spec2d (en iyi derin) | 0.695 | 1 |
| gated | 0.689 | 2 |
| **shallow (klasik)** | **0.677** | **3 (score ile eşit)** |
| score | 0.677 | 3 (shallow ile eşit) |
| late | 0.675 | 5 |
| raw1d | 0.651 | 9 |
| raw1d_wide | 0.630 | 10 |
| logvar (tek öznitelik/kanal) | 0.627 | 11 |

`shallow`, 11 modelin 3.'sü (score ile eşit) ve en iyi derin modelden yalnızca 0.018
geride, birçok derin modeli (raw1d, raw1d_wide, hatta bazı füzyon operatörlerinin
altındaki sıralarını) geçiyor. Log loss'ta ise klasik modeller de iyi ama en iyisi
değil: `score` 0.501 ile birinci, logvar 0.609 ikinci, shallow 0.620 üçüncü, en iyi
"gerçek" derin model (gated) 0.655. Yani klasik modeller yalnızca rekabetçi değil,
iyi kalibre de; ama kalibrasyonda asıl birinci `score` füzyonu (bkz. §4).

Bonn'daki kadar uç değil (orada tek öznitelik neredeyse tavana çıkıyordu), ama yön
aynı: **bu ölçütte de derin öğrenmenin kattığı marj küçük.** Bu, projenin ana
tezini ikinci bağımsız bir veri setinde ikinci bir açıdan destekliyor.

## 6. Genel makaleye etkisi

1. **Ana iddia güçlendi, iki koşulda da ayakta:** operatörler ayırt edilemiyor, hem
   Bonn'da (denklik kanıtlanmış) hem CHB-MIT'te (denklik kanıtlanamasa da fark da yok).
2. **"Neden ayırt edilemiyor" sorusunun cevabı iki korpusta farklı:** Bonn'da
   operatörler gerçekten birbirine yakın performans veriyor (denklik güçlü). CHB-MIT'te
   deneklerarası değişkenlik operatör farkını gölgeliyor (denklik zayıf, güç yetersiz).
   Bu ayrım makalenin tartışma bölümünde net yapılmalı, ikisini aynı cümlede
   "ayırt edilemiyor" diye harmanlamak yanlış olur.
3. **Early füzyon bulgusu korpusa özgü olabilir**, evrensel bir iddia olarak
   sunulmamalı.
4. **Log loss ikinci kez kırılgan çıktı**, Brier'in birincil metrik olma gerekçesi
   güçlendi.
5. **CHB-MIT'in istatistiksel gücü düşük.** 24 denek, döngüsel LOSO ile bile,
   δ_min ~0.06-0.08 mertebesinde kalıyor. Daha fazla denek (tam CHB-MIT zaten
   kullanılıyor, bu üst sınır) veya farklı bir tasarım (örn. denek içi tekrar) olmadan
   bu daha fazla daralmaz. Bu bir sınırlılık olarak yazılmalı.

## 7. Açık iş

- Bonn'daki gibi CHB-MIT için de duyarlılık analizi yapılmadı (pencere uzunluğu,
  frekans tavanı, notch filtre, alt örnekleme oranı). `docs/11` §7'de not edildi.
- `score`, `logvar`, `shallow` artık ekli; CHB-MIT model seti Bonn ile birebir aynı
  (11 model). Bu maddeler kapandı.
- **`score`'un iç tutarlılık kontrolü ve thread bulgusunun üçüncü doğrulaması
  (22 Eylül 2026).** `chbmit_score.py`, `score`'u hesaplamak için raw1d ve spec2d'yi
  aynı seed'lerle yeniden eğitti ve sonucu ana koşudaki (`perfold_before_score.csv`)
  değerlerle karşılaştırdı. İlk bakışta alarm verici görünen bir fark bulundu, ama
  kaynağı incelemenin kendi ayrıştırma betiğindeki bir kaydırma hatası (log
  dosyasındaki raw1d/spec2d satırları bir sonraki katmanın etiketiyle
  eşleştirilmişti) olduğu ortaya çıktı. Düzeltilmiş karşılaştırma:

  | | fold 0-4 (orijinalde 12 thread) | fold 5-23 (orijinalde 6 thread) |
  |---|---|---|
  | 6 thread ile yeniden eğitilince ortalama fark | 0.052 | **0.0003** |

  Yani **6 thread ile eğitilen katmanlar, tekrar 6 thread ile eğitilince neredeyse
  birebir aynı sonucu veriyor** (determinizm doğrulandı: aynı fold, aynı model, 3
  ayrı çalıştırma, üçü de f1=0.5957, epochs_run=45, best_epoch=32, val_f1=0.7558,
  ondalık basamağına kadar özdeş). Fark yalnızca thread sayısı **değiştiğinde**
  ortaya çıkıyor (12'den 6'ya), bu da docs/10'daki bulgunun üçüncü bağımsız
  doğrulaması (Bonn'da iki kez, şimdi CHB-MIT'te bir kez daha). Tek uç değer
  (fold 4/chb05/spec2d, fark 0.208) docs/10'daki en büyük değerden (0.093) daha
  büyük, CHB-MIT'in daha büyük tensörlerinin (18 kanal, daha geniş spektrogram) bu
  etkiyi büyüttüğünü gösteriyor olabilir.

  **Sonuç:** `perfold.csv`'deki raw1d/spec2d satırları hiç değişmedi (script onları
  üzerine yazmadı). `score`'un dayandığı ara hesaplama 6 thread'te üretildi;
  fold 5-23 için bu tabloyla tam tutarlı, fold 0-4 için küçük (~0.05 ortalama) bir
  sapma var. Bu sapma `score`'un genel sonucunu (F1 0.677, ortalanan bir istatistik,
  24 katman üzerinden) anlamlı ölçüde etkilemez.

- **chb01/chb21 aynı denek sorunu (23 Eylül 2026, arka plan literatür taraması
  sırasında bulundu).** PhysioNet'in kendi CHB-MIT sayfası şunu söylüyor: "Case
  chb21 was obtained 1.5 years after case chb01, from the same female subject."
  chb24 de `SUBJECT-INFO`'da yok. Yani 24 klasör var ama en az 22-23 farklı birey.
  `src/chbmit_corpus.py: all_subjects()` klasör adına göre grupluyor, chb01 ve
  chb21'i ayrı denek sayıyor. Sonuç: LOSO'da chb01 test edilirken chb21 (aynı kişi)
  eğitimde, ve tam tersi. Etki `results_v2/chbmit/loso_main/perfold.csv` üzerinden
  doğrudan ölçülebiliyor ve tek yönlü değil, asimetrik:

  | | chb01 fold (fold 0) | chb21 fold (fold 20) | 24 fold ortalaması |
  |---|---|---|---|
  | late f1_macro | 0.981 | 0.444 | 0.675 |
  | gated f1_macro | 0.962 | 0.444 | 0.689 |

  chb01 fold'u tüm modellerde ortalamanın belirgin üzerinde (sızıntıyla tutarlı),
  chb21 fold'u ise çoğu modelde en kötü fold'lardan biri (muhtemelen chb21'in kendi
  kaydının, 1.5 yıl sonraki farklı beyin olgunluğu/gürültü karakteriyle, zaten zor
  bir kayıt olması). Sadece chb01 fold'unu çıkarınca her modelin ortalama F1'i
  0.006-0.013 düşüyor (spec2d_wide istisna, +0.002), sıralama neredeyse aynı kalıyor
  (Spearman rho=0.88, p=0.0003, tam liste vs chb01 hariç liste arasında). Yani
  **etki küçük ve makalenin ana bulgusunu değiştirmiyor**, ama "sızıntıya kapalı"
  iddiamızda gerçek, belgelenmiş bir istisna. `paper/manuscript.md` §5.2 ve §7'ye
  eklendi. Düzeltme (chb01+chb21'i tek denek olarak birleştirip LOSO'yu yeniden
  koşmak) henüz yapılmadı; bu, danışman gözetiminde planlanan temiz yeniden
  uygulamanın kapsamına bırakıldı.
