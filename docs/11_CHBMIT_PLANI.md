# CHB-MIT dış doğrulama planı

**Tarih:** 18 Eylül 2026
**Durum:** chb01 indiriliyor, okuma ve pencereleme katmanı yazılıyor.
**Kod:** `src/chbmit.py`

---

## 1. Neden bu veri seti

Bonn üzerinde iki şey yapılamıyor:

1. **Denek bazlı çapraz doğrulama.** Bonn denek kimliği yayımlamıyor. Bu, NeuroAtlas
   (arXiv:2605.14698) tarafından da doğrulanmış bir kısıt: "Bonn provides no per-trial
   subject identifier, so subject-grouped cross-validation is not applicable."
   Dolayısıyla "model yeni bir hastaya genelleniyor mu" sorusu sorulamıyor bile.
2. **Ayırt edici bir ölçüt.** Aynı çalışma Bonn'un doygun olduğunu, on bir modelin
   AUROC 0.99 üzerine çıktığını ve veri setinin "sanity check" olarak görülmesi
   gerektiğini yazıyor.

CHB-MIT ikisini de çözüyor: 22 denek, 23 vaka, gerçek leave-one-subject-out mümkün.

## 2. İki veri setinin karşılaştırması

| | Bonn | CHB-MIT |
|---|---|---|
| Boyut | 19 MB | 42.6 GB (tamamı), chb01 ~1.7 GB |
| Yapı | 500 kesilmiş segment | 664 EDF, sürekli kayıt, ~969 saat |
| Denek | Kimlik yok | 22 denek / 23 vaka |
| Kanal | 1 | 23 (bazı kayıtlarda 24 veya 26) |
| Örnekleme | 173.61 Hz | 256 Hz |
| Montaj | Tek kanal | Bipolar 10-20 |
| Nöbet | 100 ictal segment (dengeli) | 198 nöbet, sürenin yüzde 1'inden azı |
| Filtre | 0.53-40 Hz kayıtta uygulanmış | Ham |
| Lisans | Akademik kullanım | ODC-By v1.0, kayıt gerekmiyor |

chb01 özeti (ayrıştırılmış ve `.seizures` dosya sayısıyla çapraz doğrulanmış):
42 kayıt, 7 nöbet, toplam ictal süre 442 saniye.

| Dosya | Başlangıç (s) | Bitiş (s) | Süre (s) |
|---|---|---|---|
| chb01_03 | 2996 | 3036 | 40 |
| chb01_04 | 1467 | 1494 | 27 |
| chb01_15 | 1732 | 1772 | 40 |
| chb01_16 | 1015 | 1066 | 51 |
| chb01_18 | 1720 | 1810 | 90 |
| chb01_21 | 327 | 420 | 93 |
| chb01_26 | 1862 | 1963 | 101 |

## 3. Tasarım kararları

### 3.1 EDF okuma: kendi okuyucumuz

`mne` veya `pyedflib` kurmak yerine okuyucu `src/chbmit.py` içinde yazıldı. Gerekçe:
EDF sabit uzunluklu ASCII başlık ve int16 veri kayıtlarından oluşur, yüz satırdan kısa
okunur. Ölçek dönüşümü (dijitalden mikrovolta) ve kanal seçimi sonuçları doğrudan
etkilediği için görünür olması tercih edildi. Bağımlılık yüzeyi de küçük kalır.

Doğrulama: okunan sinyalin süresi başlıktaki kayıt sayısı ile örtüşmeli, nöbet
sayıları özet dosyası ile `.seizures` dosya sayısı arasında tutmalı, ictal pencerelerin
bant gücü interictal'den belirgin yüksek olmalı.

### 3.2 Kanallar

Kayıtlar arasında kanal kümesi değişebiliyor; bazı dosyalarda yinelenen veya boş ("-")
kanallar var. Tüm kayıtlarda ortak olan kanalların kesişimi alınır ve ilk dosyadaki
sırayla sabitlenir. Böylece model her kayıtta aynı girdiyi görür.

Bonn tek kanallı, CHB-MIT çok kanallı. Bu mimariyi değiştirir, ama **karşılaştırmayı
bozmaz**, çünkü soru her iki korpusta da "aynı bütçe altında operatörler ayrılıyor mu"
şeklinde ve tüm operatörler aynı gövdeyi paylaşıyor. İki korpusun mutlak sayıları
zaten karşılaştırılmayacak.

### 3.3 Pencere ve etiket

- 10 saniyelik pencereler. NeuroAtlas Bonn'u da 10 s'ye pencereleyerek değerlendirdiği
  için bu seçim karşılaştırılabilirlik sağlıyor.
- Etiket: pencere nöbet aralığıyla kesişiyorsa ictal.
- Sınır pencereleri: nöbet başlangıcı ve bitişine komşu pencereler belirsiz etiket
  taşır. İlk sürümde nöbet aralığıyla kesişen her pencere ictal sayılacak, duyarlılık
  kontrolü olarak sınır pencerelerini dışlayan bir varyant koşulacak.

### 3.4 Sınıf dengesizliği

chb01'de ictal süre toplam sürenin binde 3'ü. Tüm pencereler kullanılırsa hem eğitim
imkansızlaşır hem de doğruluk anlamsızlaşır.

Yaklaşım: tüm ictal pencereler alınır, non-ictal pencerelerden denek başına sınırlı ve
tohumlanmış rastgele örnek çekilir. Örnekleme oranı kaydedilir ve raporlanır, çünkü bu
oran ölçülen performansı doğrudan belirler. **Bu bir modelleme hilesi değil, açıkça
raporlanması gereken bir tasarım parametresidir.**

### 3.5 Değerlendirme

- **Denek bazlı:** leave-one-subject-out. Tek denekle (chb01) bu mümkün değil;
  şimdilik kayıt bazlı gruplu çapraz doğrulama yapılır, denek bazlı doğrulama daha çok
  denek indirildiğinde devreye girer.
- **Metrikler:** pencere düzeyinde makro F1 ve Brier (Faz 0'da Brier'in en ayırt edici
  metrik olduğu ölçüldü), artı **saatte yanlış alarm sabitken duyarlılık**. Sonuncusu
  NeuroAtlas'ın eleştirisine cevap: AUROC klinik açıdan kritik hata biçimlerini gizliyor.
- **İstatistik:** aynı düzeltilmiş testler. Denek bazlı CV'de test/eğitim oranı
  değişeceği için `corrected_ttest` çağrısındaki oran güncellenecek.

## 4. Hesaplama bütçesi

Bonn: 500 segment, tek kanal, 25 fold, 11 model, 11.7 saat.

CHB-MIT chb01: alt örneklemeden sonra kabaca 3 ile 5 bin pencere, 18 kanal, 2560 örnek.
Girdi tensörü Bonn'dan yaklaşık iki kat büyük, pencere sayısı benzer. Tek deneğin tek
koşusu birkaç saat sürer. Tüm denekler indirildiğinde alt örnekleme oranı bu bütçeye
göre seçilecek.

**Bu yüzden önce tek denekle boru hattı doğrulanıyor.** Ölçek kararı doğrulamadan sonra.

## 5. Riskler

| Risk | Karşılık |
|---|---|
| EDF okuyucu sessizce yanlış ölçekliyor | Bant gücü ve genlik aralığı kontrolü, özet dosyasıyla çapraz doğrulama |
| Kanal kesişimi çok küçük çıkarsa | Kesişim boyutu raporlanır; gerekirse kanal alt kümesi elle sabitlenir |
| Alt örnekleme sonucu belirliyor | Oran kaydedilir, en az iki oranla duyarlılık koşusu |
| CPU yetmez | Denek sayısı ve pencere sayısı bütçeye göre seçilir, karar belgelenir |
| Tek denekten genelleme | Tek denekle genelleme iddiası yapılmayacak; bu aşama yalnızca boru hattı doğrulaması |

## 6. chb01 doğrulama sonuçları (18 Eylül 2026)

İndirme: 42/42 EDF, 1.7 GB, boyut doğrulaması hatasız. PhysioNet'e doğrudan bağlantı
dakikada 1.6 MB veriyordu (tahmini 18 saat); açık S3 aynası
(`physionet-open.s3.amazonaws.com`) 2.8 MB/s verdi ve indirme dakikalar sürdü.

**Üç bağımsız doğrulama geçildi:**

1. **Anotasyon sayımı.** Özet dosyasından 7 nöbet ayrıştırıldı, `.seizures` dosya sayısı
   ve literatürdeki chb01 değeri ile uyuşuyor.
2. **Fizyolojik kontrol.** chb01_03'te nöbet dönemi ile 20 rastgele nöbetsiz dönem
   karşılaştırıldı: delta gücü 7.7 kat, theta 6.5 kat, beta 4.8 kat, gama 18.3 kat,
   RMS genlik 2.6 kat yüksek. Bu, nöbetin bilinen EEG imzasıdır. Anotasyon hizalaması
   veya ölçek dönüşümü hatalı olsaydı bu tablo çıkmazdı.
3. **Pencereleme.** Nöbet 2996-3036 s. Sınır pencereleri dahil 5, tam içeride kalanlar
   3 pencere üretildi; elle hesapla birebir aynı.

**Ölçüm:** 42 kayıt, 40.5 saat, 22 ortak kanal (T8-P8 bazı kayıtlarda yinelendiği için
23'ten 22'ye düştü), 10 s pencerede 14.598 pencere, bunların **50'si ictal (binde 3.4)**.
İşlem süresi 7.8 saniye, bellek 56 MB.

### 6.1 Asıl kısıt disk veya CPU değil, ictal pencere sayısı

Tek denek 40 saat kayıt içeriyor ama yalnızca **50 ictal pencere** veriyor. Bonn'da 100
ictal segment vardı. Yani tek denekle eğitim kümesi Bonn'dan küçük.

Pencere kaydırmayla sayı artırılabilir:

| Adım (stride) | chb01 ictal pencere |
|---|---|
| 10 s (örtüşmesiz) | 43 |
| 5 s | 80 |
| 2 s | 191 |
| 1 s | 379 |

**Ama bu tehlikeli.** 2 saniye kaydırmayla üretilen pencereler aynı nöbetin neredeyse
aynı parçalarıdır. Aynı nöbetin pencereleri hem eğitimde hem testte yer alırsa model
ezberlediğini genelleme sanır ve doğruluk şişer. Bu, CHB-MIT literatüründe belgelenmiş
bir hata sınıfıdır (Ali ve ark. 2024, *Royal Society Open Science* 11:230601, doi:
10.1098/rsos.230601).

**Kural:** bölmeler pencere düzeyinde değil, **nöbet ve denek düzeyinde** yapılmalı.
Aynı nöbetin tüm pencereleri aynı katmanda kalmalı. Örtüşmeli pencereleme ancak bu kural
uygulanırsa kullanılabilir, ve kullanılırsa örtüşmesiz varyantla birlikte raporlanmalı.

### 6.2 Ölçek tahmini

22 denek, 198 nöbet: örtüşmesiz 10 s pencerede kabaca **1400 ictal pencere** ve gerçek
leave-one-subject-out için 22 katman. Tek denekte ne denek bazlı doğrulama mümkün, ne de
yeterli ictal örnek var. Boru hattı doğrulandığına göre darboğaz artık veri miktarıdır.

## 7. Kaynak veride bulunan anotasyon hatası

Tüm deneklerin özet dosyaları ayrıştırılıp nöbet sayıları toplandığında 198 çıktı, bu
PhysioNet'in belgelediği sayıyla birebir aynı. Ancak toplam ictal süre 17.856 saniye
çıktı ve chb09 satırı tuhaftı: 4 nöbetten 6.521 saniye, yani nöbet başına 27 dakika.

Kaynak dosyaya bakıldığında `chb09-summary.txt` içinde şu görüldü:

```
File Name: chb09_08.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 2951 seconds
Seizure 1 End Time:   3030 seconds
Seizure 2 Start Time: 9196 seconds
Seizure 1 End Time:   9267 seconds     <- "Seizure 2 End Time" olmalıydı
```

İkinci nöbetin bitiş satırı yanlış numaralandırılmış. **Bu bizim kodumuzun değil,
veri setinin hatası.** Nöbet numarasına göre eşleştiren bir ayrıştırıcı 2951-9267
arasını tek nöbet sayar: 6.316 saniyelik, var olmayan bir nöbet. Etiketlemede
kullanılsaydı yaklaşık 630 nöbetsiz pencere ictal işaretlenecek, model gürültüyü nöbet
sanmayı öğrenecekti.

**Düzeltme:** ayrıştırma sıra tabanlı hale getirildi. Her "Start Time" yeni bir nöbet
açar, her "End Time" en son açılanı kapatır; numara hiç okunmaz. Ayrıca her blokta
ayrıştırılan nöbet sayısı "Number of Seizures in File" beyanıyla karşılaştırılır ve
uyuşmazlıkta uyarı basılır.

**Düzeltme sonrası:**

| | Önce | Sonra |
|---|---|---|
| Nöbet sayısı | 198 | 198 |
| Toplam ictal süre | 17.856 s | 11.611 s |
| Uyarı | yok (hata sessizdi) | yok (24 deneğin hepsi beyanla tutarlı) |

Tek bir yazım hatası toplam nöbet süresini yüzde 54 şişiriyordu. Düzeltilmiş değerler
tutarlı: en uzun nöbet 752 s (chb11_99), en kısa 6 s (chb16), ikisi de CHB-MIT'te
bilinen değerler. Örtüşmesiz 10 s pencerede korpus genelinde yaklaşık **1161 ictal
pencere** bulunuyor.

Bu bulgu makalede raporlanmalı. Ayrıca `src/verify_chbmit.py` bu tür sorunları
indirmeden sonra otomatik denetler: dosya boyutu başlıktan hesaplanan beklenen boyutla,
nöbet aralıkları kayıt süresiyle, özet dosyası `.seizures` dosyalarıyla karşılaştırılır.

## 8. Tam korpus indirme ve doğrulama (18 Eylül 2026)

İndirme: 686 EDF, 43 GB, S3 aynasından. Dosya listesi resmi `RECORDS` dosyasından
alındı, elle yazılmadı.

`python -m src.verify_chbmit` sonucu: **686/686 dosya sağlam, 24 denek, 982.9 saat,
sorun yok.**

Yol boyunca çıkan dört ayrı sorun ve çözümleri:

### 8.1 Dosya adındaki artı işareti

`chb02/chb02_16+.edf` indirilemedi. Sebep: URL'de "+" karakteri boşluk olarak
yorumlanıyor. `%2B` ile kodlanarak çözüldü. Dosya tam boyutta indi.

### 8.2 Resmi indeks dosyasında hata

`RECORDS-WITH-SEIZURES` nöbetli kayıtlar arasında `chb07/chb07_18.edf` listeliyor.
Ancak:

- `chb07-summary.txt`, chb07_18 için "Number of Seizures in File: 0" diyor
- chb07_18 için sunucuda `.seizures` dosyası yok (HTTP 404)
- chb07_19 için hem özet bir nöbet bildiriyor hem de `.seizures` dosyası var

İki bağımsız kaynak chb07_19'da anlaşıyor, resmi indeks dosyası hatalı. Üç kaynağın
karşılaştırması yapıldı: özet dosyaları ile `.seizures` dosyaları **141 kaydın
141'inde** birebir uyuşuyor, tek sapma indeks dosyasındaki bu kayıt.

### 8.3 Sıfır dolgulu dosya

`chb17b_69.edf` başlıktan hesaplanan beklenen boyuttan 256 bayt büyük. Sunucudaki
`Content-Length` de aynı ve fazlalığın tamamı sıfır, yani eksik indirme değil,
kaynağın özelliği. Okuyucu fazlalığı zaten yok sayıyor. Doğrulayıcı artık kesik
dosya (boyut < beklenen) ile sıfır dolgu (boyut > beklenen, fazlalık sıfır) ayrımını
yapıyor ve yalnızca ilkini sorun sayıyor.

### 8.4 Kanal montajı: naif kesişim çöküyor

CHB-MIT'te kanal kümesi denekler arasında ve **aynı deneğin kayıtları arasında**
değişiyor. Ham kanal sayıları 22 ile 38 arasında. En kritik durum chb12: kayıtlarının
bir bölümü CS2 referanslı bir montajla alınmış (C3-CS2, CP2-CS2 gibi) ve standart
bipolar montajla **tek bir kanalı bile paylaşmıyor**. Tüm kayıtların kesişimi bu
yüzden boş küme.

Naif kesişim bunu sessizce yapıyordu: chb12 için 0 kanal, korpus geneli için 0 kanal.
Hata vermeden sıfır kanal döndüğü için fark edilmesi zordu.

**Çözüm:** literatürde kullanılan standart 18 kanallı bipolar montaj sabit hedef
olarak tanımlandı (`TARGET_CHANNELS`), bu kanalların tamamını içermeyen kayıtlar
elenir ve elenen kayıt/nöbet sayısı raporlanır.

| | Korunan | Toplam |
|---|---|---|
| Kayıt | 683 | 686 |
| Nöbet | 185 | 198 |
| Denek | 24 | 24 |

Düşen 3 kayıt ve 13 nöbetin tamamı chb12'nin farklı montajlı bölümünden.

### 8.5 chb17 tamamen atlanıyordu

`subject_files` kalıbı `{denek}_*.edf` idi. chb17'nin kayıtları `chb17a_03.edf`,
`chb17b_69.edf`, `chb17c_...` biçiminde adlandırılmış, yani alt çizgi doğrudan
denek adından sonra gelmiyor. Kalıp bu deneğin **21 kaydının tamamını sessizce
atlıyordu** ve hata vermiyordu. Kalıp `{denek}*.edf` olarak düzeltildi.

Bu ve 8.4, aynı sınıftan hatalar: veri yükleme katmanı sessizce boş küme döndürüyor
ve boru hattı çalışmaya devam ediyor. Temiz kurulumda yükleyicinin beklenen kayıt ve
kanal sayısını doğrulaması, uyuşmazlıkta hata vermesi gerekir.

## 9. Sızıntıya kapalı bölme katmanı

**Kod:** `src/chbmit_corpus.py`

### 9.1 İki kural

1. Aynı **deneğin** hiçbir penceresi hem eğitimde hem testte olamaz.
2. Aynı **nöbetin** hiçbir penceresi bölünemez.

İkinci kural birincisi uygulandığında kendiliğinden sağlanır, ama örtüşmeli
pencereleme veya denek içi bölme denenirse ayrıca gerekir. Bu yüzden her pencere
nöbet kimliğini (`chb09_08.edf#2` gibi) veriyle birlikte taşır ve denetim ikisini de
ayrı ayrı kontrol eder.

### 9.2 Bölme yöntemleri

- `leave_one_subject_out`: her denek sırayla test kümesi olur. Denek bazlı
  genellemenin asıl ölçüsü budur ve Bonn'da yapılamayan şey tam olarak budur.
- `grouped_kfold_by_subject`: LOSO pahalıysa kullanılır. Denekler pencere sayısına
  göre sıralanıp sırayla en az yüklü katmana atanır, böylece katmanlar dengelenir.
  İstenen katman sayısı denek sayısını aşarsa **hata verir**, sessizce boş katman
  üretmez.

### 9.3 Alt örnekleme denek içinde yapılır

Non-ictal pencereler her deneğin **kendi** ictal sayısına oranla tutulur. Korpus
genelinde tek bir oran kullanılsaydı, nöbeti çok olan denekler (chb12'de 27, chb24'te
16) diğerlerinin non-ictal örneklerini bastırırdı.

### 9.4 Denetimin kendisi sınandı

Bir sızıntı denetiminin çalıştığını göstermenin tek yolu, onu başarısız olmaya
zorlamaktır. `--check` bu yüzden kasten sızdıran bir bölme de dener: pencereler
denek ve nöbet gözetilmeden rastgele ikiye ayrılır.

Sonuç: denetim bu yapay sızıntıyı yakaladı ve **iki kuralı da ayrı ayrı** bildirdi
(dört deneğin dördü hem eğitimde hem testte, ve nöbetler bölünmüş). Gerçek
bölmelerde (LOSO ve gruplu k-kat) hiçbir sorun bulunmadı.

### 9.5 Yükleyici artık sessizce boş dönmüyor

`validate()` her kurulumda şunları denetler ve uyuşmazlıkta hata fırlatır: dizi
uzunluklarının tutarlılığı, kanal sayısı, pencere uzunluğu, sonlu olmayan değer,
en az bir ictal pencere, her ictal pencerenin bir nöbet kimliği taşıması ve hiçbir
non-ictal pencerenin taşımaması.

Gerekçe bölüm 8.4 ve 8.5'teki iki hata: ikisi de hata vermeden sıfır döndürüyordu.
Bu projede en pahalı hata sınıfı bu, çünkü boru hattı çalışmaya devam ediyor ve
sonuç yalnızca "biraz düşük" görünüyor.

## 10. Korpus, ön işleme ve hesaplama bütçesi

### 10.1 Kurulan korpus

```
6380 pencere, 1276 ictal, 185 nöbet, 24 denek, 18 kanal x 2560 örnek (1.18 GB)
LOSO 24 katman ve 5 katlı gruplu bölme: sızıntı yok
Yapay sızıntı testi: denetim yakaladı
```

Katman dengesi (5 katlı gruplu bölme): katman başına 246 ile 265 ictal pencere.

Denekler arası dengesizlik dikkate alınmalı: chb15'te 214 ictal pencere varken
chb16'da 14 var. LOSO'da chb16 test katmanı olduğunda test kümesinde yalnızca 14
ictal pencere bulunur; denek bazlı sonuçlar bu yüzden denek başına ayrı ayrı
raporlanmalı, tek bir ortalamaya indirgenmemeli.

### 10.2 Bonn'dan ayrılan iki ön işleme kararı

**Frekans tavanı 40 Hz'den 64 Hz'e çıkarıldı.** Bonn kayıt sırasında 0.53-40 Hz'e
sınırlıydı, orada 40 Hz tavan bilgi kaybı değildi. CHB-MIT band sınırlı değil ve bu
projede yapılan ölçüm gama bandında (30-70 Hz) ictal/interictal güç oranını 18.3 kat
buldu; delta 7.7, theta 6.5, beta 4.8 katta kalıyordu. 40 Hz'de kesmek en ayırt
edici bandı atmak olurdu.

**Şebeke gürültüsü bir duyarlılık ekseni.** CHB-MIT Boston'da kaydedilmiştir, şebeke
60 Hz ve bu yeni tavanın içinde kalıyor. Gürültü sabit olduğu için ictal/interictal
oranını tek başına açıklayamaz, ama spektrogramda güçlü bir şerit oluşturur.
`notch_hz` ile bastırılabilir, varsayılan kapalı, iki türlü de koşulmalı.

**Normalizasyon seçenekleri** (`src/chbmit_prep.py`): `none`, `window` (varsayılan,
her pencere her kanal kendi içinde), `channel` (kanal başına, istatistik **yalnızca
eğitim katmanından**). Sonuncusu için fonksiyon eğitim maskesi ister; test
istatistiğinin kullanılması sızıntıdır ve bu tür normalizasyon sızıntısı literatürde
sık görülür.

### 10.3 Modellerin çok kanallı girdiye uyarlanması

Kodlayıcılarda giriş kanal sayısı 1'e sabitlenmişti, parametreye çevrildi. Bonn
parametre sayıları birebir korundu (raw1d 27.075, late 98.571, early 97.507), yani
eski sonuçlarla uyumluluk bozulmadı.

18 kanalla parametre eşlemesi de korunuyor:

| Model | Bonn (1 kanal) | CHB-MIT (18 kanal) |
|---|---|---|
| late | 98.571 | 102.768 |
| gated | 98.698 | 102.921 |
| attention | 98.544 | 102.767 |
| early | 97.507 | 102.547 |
| spec2d_wide | 97.775 | 102.236 |
| raw1d_wide | 99.795 | 103.712 |

Füzyon operatörleri arasındaki yayılım yüzde 0.36, kapasite eşlenmiş tek modaliteli
kontroller de aynı bantta. Karşılaştırmanın temel kısıtı yeni korpusta da geçerli.

### 10.4 Ölçülen hesaplama maliyeti

12 thread, 5055 pencerelik eğitim katmanı, epoch başına: late 9.4 s, early 6.6 s,
raw1d 5.6 s. Ortalama 7.2 s kabul edilip erken durdurmanın 30 epoch civarında
devreye girdiği varsayılırsa:

| Protokol | Eğitim sayısı | Süre | Eşli ölçüm |
|---|---|---|---|
| 5 kat gruplu x 5 tekrar | 275 | ~16.5 saat | n=25 |
| 5 kat gruplu x 3 tekrar | 165 | ~9.9 saat | n=15 |
| 5 kat gruplu x 1 | 55 | ~3.3 saat | n=5 |
| **LOSO 24 katman** | **264** | **~15.8 saat** | **n=24** |

**Öneri: LOSO.** Aynı maliyete 5x5 tekrarlı kurgu kadar eşli ölçüm veriyor (24'e 25),
ama her katman gerçek bir hastanın dışarıda bırakılması olduğu için bilimsel olarak
doğru protokol bu. Ayrıca tekrar kaynaklı ek korelasyon sorunu doğmuyor; katmanlar
eğitim verisini paylaştığı için düzeltilmiş test yine gerekli, oran 1/(n-1)=1/23.

## 11. Sıradaki adımlar

1. chb01 indirmesi bitsin (S3 aynasından, boyut doğrulamalı)
2. EDF okuyucu doğrulaması: süre, genlik, ictal-interictal bant gücü farkı
3. Kanal kesişimi ve pencereleme
4. Küçük ölçekli deneme koşusu, tek operatörle, boru hattının uçtan uca çalıştığını görmek
5. Kapsam kararı: kaç denek indirilecek, alt örnekleme oranı ne olacak
