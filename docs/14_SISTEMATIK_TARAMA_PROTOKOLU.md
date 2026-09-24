# Sistematik Literatür Tarama Protokolü

**Tarih:** 21 Eylül 2026
**Durum:** Taslak, aramalar henüz çalıştırılmadı.
**Neden gerekli:** `docs/09_LITERATUR_TARAMASI.md` bir kapsam belirleme (scoping)
taramasıydı, kayıtlı değildi. `paper/proposal.md` §3 açıkça "novelty iddiasından önce
kayıtlı bir tarama yapacağım" diyor. Bu belge o taramanın protokolü.

**Erişim notu:** Scopus ve Web of Science kurumsal erişim gerektirir. Bu belgedeki
sorgu dizeleri veritabanlarına elle girilip sonuçlar (RIS, CSV veya BibTeX) dışa
aktarılacak; tekilleştirme, tarama tablosu ve PRISMA akış şeması bu dışa aktarımlar
üzerinden hazırlanacak.

---

## 1. Araştırma sorusu (protokolün gerekçesi)

`paper/proposal.md` §2'deki soru: *Mimari boyutu, eğitim protokolü ve veri sabit
tutulduğunda, füzyon operatörü seçimi EEG nöbet tespitinde ölçülebilir ve
istatistiksel olarak savunulabilir bir fark yaratıyor mu?*

Taramanın amacı bu soruyu **daha önce kimin, nasıl** sorduğunu tam olarak haritalamak.
Üç seviyeli bir hedef var:

1. **Geniş harita:** EEG nöbet tespitinde çok modlu (ham sinyal + spektrogram) derin
   öğrenme füzyonunu konu alan tüm çalışmalar.
2. **Orta katman:** Bunların içinde birden fazla füzyon operatörünü karşılaştıranlar.
3. **En yakın rakip:** Parametre eşlemesi yapıp istatistiksel test kullananlar
   (şu an bilinen tek aday: Daşdemir 2025, ve o bile nöbet *tahmini* yapıyor,
   *tespit* değil).

Tablo bu üç katmanı ayırt edecek şekilde doldurulacak.

## 2. Veritabanları

| Veritabanı | Kapsam | Erişim |
|---|---|---|
| **Scopus** | Birincil | Üniversite aboneliği |
| **Web of Science** | Birincil | Üniversite aboneliği |
| **IEEE Xplore** | İkincil | Füzyon/CNN çalışmalarının büyük kısmı IEEE'de |
| **arXiv (cs.LG, eess.SP)** | Tamamlayıcı, ayrı etiketli | Hakemsiz olabilir, ayrı işaretlenir |

arXiv ayrı tutulma sebebi: `docs/09`'daki adayların çoğu (NeuroAtlas, HARMES
karşılaştırması, Brain Sciences füzyon kıyaslaması) 2026 tarihli ve Scopus/WoS'a henüz
girmemiş olabilir. Bu alan hızlı hareket ediyor, sadece hakemli veritabanlarına
güvenmek güncel çalışmaları kaçırır. Ama hakemsiz olduğu için tabloda ayrı sütunla
işaretlenecek.

## 3. Arama dizeleri

Sorgu bilerek **geniş** tutuldu. "Parametre eşlemesi", "istatistiksel test" gibi
ölçütler sorguya gömülmedi, çünkü makaleler bunu başlık/özette bu kelimelerle
anlatmaz. Bu ölçütler tam metin taramasında elle uygulanacak (bkz. §5).

### Scopus

```
TITLE-ABS-KEY(
  ("EEG" OR "electroencephalogra*")
  AND ("seizure" OR "epilep*")
  AND ("fusion" OR "multimodal" OR "multi-modal")
  AND ("deep learning" OR "CNN" OR "convolutional neural network" OR "neural network")
)
AND ( LIMIT-TO ( LANGUAGE,"English" ) )
```

### Web of Science

```
TS=(
  ("EEG" OR "electroencephalogra*")
  AND ("seizure" OR "epilep*")
  AND ("fusion" OR "multimodal" OR "multi-modal")
  AND ("deep learning" OR "CNN" OR "convolutional neural network" OR "neural network")
)
```
Dil filtresi: arama arayüzünde "English" seçilir.

### IEEE Xplore

```
("EEG" OR "electroencephalogram") AND ("seizure" OR "epilepsy" OR "epileptic")
AND ("fusion" OR "multimodal" OR "multi-modal")
AND ("deep learning" OR "CNN" OR "convolutional neural network")
```
(Full Text & Metadata alanında; IEEE arayüzü Scopus/WoS'unkinden farklı sözdizimi
kullanabilir, arayüzdeki "Command Search" seçeneği yukarıdakine en yakın sonucu verir.)

### arXiv (tamamlayıcı, elle)

`arxiv.org/list/eess.SP/recent` ve `cs.LG` kategorilerinde, veya
`arxiv.org/abs` arama kutusunda:

```
abs:"EEG" AND abs:"seizure" AND abs:(fusion OR multimodal) AND abs:("deep learning" OR CNN)
```

Tarih aralığı önerisi: alt sınır yok (2015 öncesi CNN tabanlı EEG çalışması pratikte
yok), üst sınır arama tarihi. Yayın türü: makale + konferans bildirisi, derleme
(review) hariç tutulmaz ama ayrı bir kovada tutulur (bkz. §4).

## 4. Dahil etme / hariç tutma ölçütleri

### Dahil

1. EEG sinyali kullanıyor
2. Nöbet tespiti **veya** tahmini konusu (ikisi de dahil, tabloda ayrı etiketlenir,
   çünkü en yakın rakip Daşdemir 2025 tahmin tarafında)
3. En az iki farklı temsili (ham sinyal + spektrogram, veya iki farklı modalite)
   derin öğrenmeyle birleştiriyor
4. İngilizce
5. Makale, konferans bildirisi veya (ayrı etiketli) arXiv ön baskısı

### Hariç

1. EEG/nöbet dışı bir alan
2. Tek modaliteli (füzyon yok)
3. Yalnızca derleme/gözden geçirme makalesi → ana karşılaştırma tablosuna girmez,
   ayrı "derleme" kovasına konur (Roy 2019, Shoeibi 2021, Xu 2024, Guarrasi 2025 gibi;
   bunlar bağlam için tutulur, karşılaştırma satırı olarak sayılmaz)
4. Tam metne erişilemiyor (üniversite aboneliğiyle de)

## 5. Tarama süreci (iki aşamalı)

**Aşama 1: başlık ve özet taraması.** Her kayıt için üç etiketten biri: `dahil`,
`hariç`, `belirsiz`. Hariç tutma gerekçesi kaydedilir (§4'teki numara).

**Aşama 2: tam metin taraması.** `dahil` ve `belirsiz` olanların tam metni okunur,
§6'daki veri çıkarma tablosu doldurulur.

**Tek kişilik tarama sorunu:** PRISMA standart olarak iki bağımsız değerlendirici
önerir, yanlılığı azaltmak için. Bu proje tek öğrenci tarafından yürütülüyor. Bunu
gizlemek yerine açıkça sınırlılık olarak yazılmalı. Kısmi düzeltme: rastgele seçilen
%15-20'lik bir alt küme (Aşama 1 sonrası) ikinci bir gözle (danışman, akran veya
belgelenmiş şekilde AI destekli ikinci geçiş) tekrar taranıp uyuşma oranı raporlanır.
Bu ikinci geçiş insan değerlendirici yerine geçmez, bu netçe belirtilmeli.

## 6. Veri çıkarma tablosu (şema)

Her `dahil` kayıt için doldurulacak sütunlar:

| Sütun | Açıklama |
|---|---|
| `ref` | Yazar, yıl |
| `doi` | Crossref API ile doğrulanacak |
| `venue` | Dergi/konferans, arXiv ise açıkça "arXiv (hakemsiz)" |
| `task` | tespit / tahmin |
| `dataset` | Bonn, CHB-MIT, TUSZ, diğer |
| `n_operators` | Karşılaştırılan füzyon operatörü sayısı |
| `operators` | İsimleri (early, late, gated, attention, score, ...) |
| `param_matched` | Evet / Hayır / Belirtilmemiş |
| `stat_test` | Yok / Naif (düzeltmesiz) / Düzeltilmiş / Diğer |
| `cv_design` | Kaç kat, kaç tekrar, hangi bölme (subject-wise mi?) |
| `gap_claimed` | Yazarların kendi boşluk iddiası (varsa) |
| `notes` | Serbest not |

Bu şema `docs/09`'da kullanılan tabloyla uyumlu; oradaki 15 çalışma bu şemaya
aktarılıp taban olarak kullanılabilir, sıfırdan başlanmaz.

## 7. PRISMA akış şeması (dolduruimak üzere iskelet)

```
Belirleme (Identification)
  Scopus'tan gelen kayıt: n = ____
  Web of Science'tan gelen kayıt: n = ____
  IEEE Xplore'dan gelen kayıt: n = ____
  arXiv'den gelen kayıt (ayrı havuz): n = ____
  Tekilleştirme sonrası: n = ____

Tarama (Screening)
  Başlık/özet taranan: n = ____
  Bu aşamada hariç tutulan: n = ____ (gerekçe dağılımı §4 numaralarına göre)

Uygunluk (Eligibility)
  Tam metni değerlendirilen: n = ____
  Tam metinde hariç tutulan: n = ____ (gerekçe dağılımı)

Dahil edilen (Included)
  Ana karşılaştırma tablosunda: n = ____
  Ayrı derleme kovasında: n = ____
```

Bu şema doldurulduktan sonra makalenin yöntem bölümüne bir şekil olarak girecek.

## 8. Zaman çizelgesi

`paper/proposal.md`'deki takvimle uyumlu (Güz 2026, 1-4. haftalar):

| Hafta | İş |
|---|---|
| 1 | Aramaları çalıştır, dışa aktar, tekilleştir |
| 2 | Aşama 1: başlık/özet taraması |
| 3 | Aşama 2: tam metin taraması, veri çıkarma tablosu |
| 4 | PRISMA şeması, karşılaştırma tablosu, `docs/09`'un bu belgeyle birleştirilmesi |

## 9. Sıradaki adım

1. Üniversite kütüphane erişimiyle Scopus'ta §3'teki sorgu çalıştırılır, sonuçlar RIS
   veya CSV olarak dışa aktarılır.
2. Aynısı Web of Science'ta tekrarlanır.
3. İki dışa aktarım birleştirilip tekilleştirilir ve Aşama 1 taramasına geçilir.

Protokolün tarama başlamadan önce OSF gibi bir platformda ön kaydı zorunlu değildir,
ancak makale hedefi düşünüldüğünde değerlendirilebilir.
