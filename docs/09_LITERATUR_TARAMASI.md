# Literatür Taraması: Boşluk Kanıtı

**Tarih:** 17 Eylül 2026
**Kapsam:** Kapsam belirleme (scoping) taraması. Bu **kayıtlı sistematik tarama değildir.**
Scopus ve Web of Science üzerinden PRISMA akışlı resmi tarama Faz 1'de yapılacak. Burada
amaç, boşluk iddiasının ayakta durup durmadığını erkenden sınamak.

**Yöntem:** Web araması (9 sorgu) ve bulunan adayların tam metin veya özet incelemesi.
Her künye Crossref API ile DOI üzerinden doğrulandı. Erişilemeyen iki ScienceDirect
makalesi için özet ikincil kaynaktan alındı ve bu belgede işaretlendi.

---

## 1. Sonuç özeti

**Boşluk daraldı ama ayakta.** Bu tarama, iddiayı olduğu gibi bırakmıyor, yeniden
tanımlanmasını gerektiriyor.

Ayakta kalan boşluk şu:

> EEG nöbet **tespiti** için, **üç veya daha fazla** füzyon operatörünü **eşlenmiş
> parametre bütçesi** altında karşılaştıran ve farkları **eşli istatistiksel testler ile
> denklik testi** kullanarak değerlendiren bir çalışma bulunamadı.

Bu ifadedeki her koşul gerekli, çünkü her birini tek tek karşılayan çalışmalar var.

---

## 2. En yakın çalışmalar

### 2.1 En yakın tehdit: Daşdemir 2025

**Daşdemir, A. (2025).** Epileptic seizure prediction with deep learning-based fusion
methods. *Engineering Science and Technology, an International Journal*, 72:102212.
doi:10.1016/j.jestch.2025.102212
*(Künye Crossref ile doğrulandı. Tam metne erişilemedi, özet ikincil kaynaktan.)*

Bu çalışma kendisini şöyle tanımlıyor: "a fair, protocol-controlled comparison of two
fusion strategies for EEG-based seizure prediction". Yani **adil ve protokol kontrollü
karşılaştırma** iddiası taşıyor ve bizim tasarımımıza en yakın olan bu.

Örtüşen yanlar:

- Aynı iki kodlayıcı (ham EEG dalı ve STFT tabanlı 2B dal) sabit tutulmuş
- Sadece füzyon noktası değiştirilmiş
- Adil karşılaştırma niyeti açıkça belirtilmiş

Ayrışan yanlar:

| Boyut | Daşdemir 2025 | Bu proje |
|---|---|---|
| Görev | Nöbet **tahmini** (prediction) | Nöbet **tespiti** (detection) |
| Operatör sayısı | 2 (karar düzeyi, öznitelik düzeyi) | 5 (early, late, gated, attention, score) |
| Parametre eşleme | Belirtilmemiş | Açık kısıt |
| İstatistiksel test | Raporlanmamış | Eşli test, çoklu karşılaştırma düzeltmesi, denklik |
| Kapasite eşlenmiş tek modaliteli referans | Yok | Var |

Raporlanan sonuç: karar düzeyi 97.50, öznitelik düzeyi 97.70 doğruluk. **0.20 puanlık bu
fark istatistiksel test olmadan sunuluyor.** Bu, projenin işaret ettiği sorunun birebir
örneği: iki yöntem arasındaki küçük bir fark, belirsizlik ölçülmeden bir tercih gerekçesi
olarak sunuluyor.

**Eylem:** Tam metne erişilmeli. Parametre sayılarını eşlemiş ve test yapmışsa, projenin
katkısı operatör sayısına ve görev tanımına daralır. Yapmamışsa boşluk iddiası güçlenir.

### 2.2 Aynı tasarım, başka alan: HARMES karşılaştırması

**Mohamady, A., Burchard, R., Van Laerhoven, K. (2026).** A Comparison of Fusion Techniques
for Multi-Modal Human Activity Recognition on the HARMES Dataset. arXiv:2606.27886.

Yedi füzyon yöntemini ortak bir veri setinde karşılaştırıyor, katılımcı dışı bırakmalı
değerlendirme kullanıyor. Kendi boşluk iddiası şu:

> "to the best of our knowledge, no head-to-head comparison of these paradigms exists on a
> common multi-modal HAR benchmark dataset."

Bu bizim için iki yönden önemli:

1. **Tasarımın kendisi meşru ve yayımlanabilir.** Haziran 2026'da, insan aktivitesi tanıma
   alanında, "bu karşılaştırma henüz yapılmadı" gerekçesiyle bir makale çıkmış.
2. **O alanda 2026'da hâlâ yeni sayılıyorsa**, EEG nöbet tespitinde de yapılmamış olması
   şaşırtıcı değil.

Bulguları da ilgi çekici: en basit mekanizmalar en güçlüsü çıkmış (gated 0.827 makro F1,
geç birleştirme hemen arkasında), dikkat ve tensör tabanlı ağır yöntemler geride kalmış.
Ancak özet metninde **parametre eşlemesinden ve istatistiksel testten söz edilmiyor.**

### 2.3 Kontrollü karşılaştırma, ama EEG değil

**Benchmarking Multimodal Deep Fusion Strategies for Heterogeneous Neuroimaging and
Cognitive Data Using a Controlled Sex Classification Task (2026).** *Brain Sciences*,
16(4):405. doi:10.3390/brainsci16040405

Erken, dikkat tabanlı, altuzay tabanlı ve çizge tabanlı füzyonu birleşik bir çerçevede
karşılaştırıyor. Erken füzyonun tutarlı biçimde kazandığını ve füzyon stratejisi ile
öznitelik ölçeklemenin, mimari karmaşıklıktan daha belirleyici olduğunu bildiriyor.

Ayrışma: veri, tablo biçiminde nörogörüntüleme ve bilişsel ölçümler, yani **zaman serisi
değil**. Parametre sayıları eşlenmemiş. 3 katlı çapraz doğrulama üzerinde bootstrap güven
aralığı var, eşli test veya denklik testi yok.

Bu çalışma projeyi baltalamıyor, aksine destekliyor: "füzyon stratejisi mimari
karmaşıklıktan önemli" bulgusu, stratejinin izole edilerek ölçülmesi gerektiğini söylüyor.

### 2.4 Kritik destek: Bonn veri seti doygun

**Kontras, K. ve ark. (2026).** NeuroAtlas: Benchmarking Foundation Models for Clinical EEG
and Brain-Computer Interfaces. arXiv:2605.14698.

42 EEG veri seti, 260 bin saat. Ek bölüm D.1.3'te Bonn hakkında şunlar yazıyor
(doğrudan alıntı, HTML tam metinden doğrulandı):

> "Bonn saturates: eleven models reach AUROC ≥ 0.99 (BIOT, REVE, STEEGFormer-L, all four
> Chronos sizes, NeuroLM, SleepFM, MOMENT-S, and CBraMod), confirming that the S-versus-rest
> split is recoverable from almost any representation. We therefore treat Bonn as a sanity
> check rather than a discriminating benchmark"

ve

> "Bonn provides no per-trial subject identifier, so subject-grouped cross-validation is not
> applicable."

**Bu iki cümle projenin iki sınırlılığını bizim iddiamız olmaktan çıkarıp literatürün
tespiti haline getiriyor.** Artık şunları yazabiliyoruz:

- Bonn'un klasik ikili görevi doygun, ayırt edici bir ölçüt değil. Bizim tek öznitelikli
  referansımızın yüksek skor alması bir anomali değil, bağımsız olarak doğrulanmış bir
  özellik.
- Denek kimliği yok, dolayısıyla denek bazlı çapraz doğrulama uygulanamaz. Bu bizim
  ihmalimiz değil, veri setinin özelliği.

Doğrudan sonuç: **ikinci bir korpus (CHB-MIT) tercih değil, zorunluluk.** Tek başına Bonn
üzerinde "operatörler ayırt edilemiyor" demek, doygun bir ölçütte hiçbir şeyin ayırt
edilemeyeceğini söylemekten öteye gitmez.

### 2.5 Diğer taranan çalışmalar

| Çalışma | Ne yapıyor | Neden boşluğu kapatmıyor |
|---|---|---|
| Frontiers Signal Processing 2026, doi:10.3389/frsip.2026.1745291 | Füzyona dört yorumlanabilir parametre ekleyen "denklem düzeyi" yeniden formülasyon, 12 model üzerinde | Operatörleri birbiriyle karşılaştırmıyor, ortak bir füzyon katmanını tüm modellere uyguluyor |
| Guarrasi, V. ve ark. (2025), *Image and Vision Computing*, doi:10.2139/ssrn.4952813 (IVC 2025) | Biyomedikal uygulamalarda ara füzyonun sistematik derlemesi, biçimsel gösterim önerisi | Derleme, deneysel karşılaştırma değil. Terminoloji için değerli |
| Wang ve ark. (2026), *Scientific Reports* 16, doi:10.1038/s41598-026-41636-7 | Zaman-frekans çapraz dikkat modeli, Bonn üzerinde 93.63 | Tek mimari öneriyor, operatör karşılaştırması yok |
| Yu ve ark. (2025), *Eng* 6(7):150, doi:10.3390/eng6070150 | İki kanallı öznitelik füzyonu ile EEG sınıflandırma | Sabit füzyon tasarımı, ablasyon yok |
| Rheude, T., Eils, R., Wild, B. (2025), arXiv:2512.22991 | Çok modlu karmaşıklığın çoğu zaman karşılığını vermediğini savunuyor | Genel çok modlu ortam, EEG nöbet tespiti değil |

---

## 3. Boşluk iddiasının güncel hali

**Yazılabilecek ifade:**

> Füzyon operatörleri EEG nöbet tespiti için sıklıkla önerilmektedir, ancak genellikle teker
> teker önerilmekte ve kapasitesi farklı temel modellerle karşılaştırılmaktadır. Bildiğimiz
> kadarıyla, üç veya daha fazla füzyon operatörünü sabit parametre bütçesi altında
> karşılaştıran ve farkları eşli istatistiksel testler ile denklik testi kullanarak
> değerlendiren bir çalışma bulunmamaktadır. En yakın çalışma [Daşdemir 2025] nöbet tahmini
> görevinde iki operatörü karşılaştırmakta, ancak parametre eşlemesi veya istatistiksel test
> raporlamamaktadır.

**Yazılamayacak ifade:**

> "Bu karşılaştırma literatürde hiç yapılmamıştır."

Çünkü hem Daşdemir 2025 kısmen yapmış, hem de tarama henüz sistematik değil.

---

## 4. Tasarıma etkileri

1. **CHB-MIT artık zorunlu.** NeuroAtlas'ın doygunluk tespiti, Bonn tek başına iken
   "operatörler ayırt edilemiyor" bulgusunu yorumlanamaz hale getiriyor.
2. **Klasik ikili görev (T3) ana sonuç olmaktan çıkarılmalı.** Doygun olduğu bağımsız
   olarak gösterilmiş. Ayırt ediciliği daha yüksek görev tanımları öne alınmalı.
3. **Daşdemir 2025 tam metni okunmalı**, ve çalışma ilgili çalışmalar bölümünde en yakın
   çalışma olarak açıkça konumlandırılmalı.
4. **Tek öznitelikli referans korunmalı.** NeuroAtlas'ın doygunluk tespitiyle aynı yönde,
   bağımsız bir teyit sağlıyor.
5. **Nöbet tahmini ile tespiti ayrımı** metinde net yapılmalı, çünkü en yakın çalışma
   tahmin tarafında.

---

## 5. Açık işler

- [ ] Daşdemir 2025 tam metin (kütüphane erişimi veya yazardan talep)
- [ ] Guarrasi 2025 tam metin, standart karşılaştırma eksikliğine dair ifadeler için
- [ ] Kayıtlı sistematik tarama: Scopus ve Web of Science, PRISMA akışı, eleme ölçütleri
- [ ] HARMES makalesi tam metin: parametre eşlemesi yapmış mı, istatistiksel test var mı

## 6. Sorgu kaydı

17 Eylül 2026 tarihinde çalıştırılan sorgular:

1. parameter-matched comparison of fusion strategies EEG seizure detection CNN
2. systematic comparison early late intermediate fusion multimodal deep learning benchmark fair
3. ablation study fusion operator 1D 2D CNN raw signal spectrogram seizure detection
4. "fusion strategies" comparison controlled equal parameters biomedical signal classification study
5. "Epileptic seizure prediction with deep learning-based fusion methods" abstract decision-level feature-level comparison
6. Bonn dataset fusion strategy comparison late fusion gated attention seizure classification equal parameter budget
7. EEG deep learning statistical significance testing repeated cross validation equivalence test seizure detection comparison models
8. "does fusion help" OR "is fusion necessary" multimodal deep learning negative result matched capacity unimodal baseline
9. Bonn EEG dataset too easy saturated benchmark criticism simple features high accuracy seizure detection
