# Gönderim hazırlığı: katkı beyanı (CRediT) ve arXiv

Taslak, 19 Ekim 2026. Katkı beyanı yazarların onayı olmadan makaleye eklenmez; önce
Emre ve Hasan kontrol eder, sonra Sevgi Hoca'ya gider (hoca CRediT'in eklenip
eklenmeyeceğine karar verecekti).

## 1. Katkı beyanı (CRediT), taslak

Rol tanımları: https://credit.niso.org (14 rol). Aşağıdaki dağılım bu projede kimin ne
yaptığına göre önerildi; yanlış ya da eksik olan yerleri düzeltin.

**Önerilen metin (İngilizce, makaleye girecek hâli):**

> **Emre Atay Tümer:** Conceptualization, Methodology, Software, Validation, Formal
> analysis, Investigation, Data curation, Visualization, Writing – original draft,
> Project administration.
> **Hasan Berk Berber:** Software (manuscript build and Overleaf bundle, literature
> screening tools), Validation (cross-machine reproducibility test on a third machine,
> checks of the reported numbers and section references), Investigation (literature
> screening), Visualization (Figure 1), Writing – review & editing.
> **Sevgi Şengül Ayan:** Conceptualization (research questions), Methodology,
> Resources, Supervision, Writing – review & editing.

Kontrol edilecek noktalar:
- **Resources (Sevgi Hoca):** Lab bilgisayarı ve GPU üniversite laboratuvarından
  sağlandıysa doğru; değilse çıkarılmalı.
- **Methodology (Sevgi Hoca):** Hocanın 10 Ekim mailindeki yöntem kararları (5 seed,
  ±0.05/±0.02 marjları, GEDAI karşılaştırması, Siena'nın rolü) bu rolü karşılıyor.
- **Zakaria dahil olursa** onun somut katkısı ayrıca yazılmalı.
- Yapay zekâ kullanımı CRediT'te bir rol değildir; ayrı beyanda (makalenin sonunda)
  zaten var.

## 2. arXiv hazırlığı

**Kategori önerisi:** birincil `eess.SP` (Signal Processing); çapraz liste `cs.LG`
(Machine Learning) ve `q-bio.NC` (Neurons and Cognition). Seçim hocanın onayıyla.

**Lisans:** arXiv gönderiminde lisans seçiliyor ve sonradan değiştirilemiyor. İki seçenek:
- `CC BY 4.0`: en açık seçenek; açık erişimli dergilerin çoğuyla uyumlu.
- arXiv'in standart lisansı (non-exclusive distribution): en kısıtlayıcı; dergi seçimi
  belirsizken en güvenli seçenek.
Karar dergi seçimine bağlı; hedef derginin ön baskı politikası kontrol edilmeli.

**Onay (endorsement):** İlk kez gönderim yapan biri `eess.SP` için arXiv'de onay
(endorsement) isteyebilir. Hocanın arXiv hesabı varsa onun gönderen yazar olması ya da
onay vermesi en kolay yol.

**Gönderimden önce yapılacaklar (kontrol listesi):**

| # | İş | Durum |
|---|---|---|
| 1 | Özeti kısaltmak: arXiv özet sınırı 1920 karakter, şu an yaklaşık 2116 karakter (316 kelime). Dergilerin çoğu da 250–300 kelime istiyor. | yapılacak |
| 2 | Giriş'teki 20 `[abstract]` işaretini tam metinlerle çözmek (literatür taraması) | literatür |
| 3 | Taslak notunu ("DRAFT v2, NOT FOR SUBMISSION") ve "(Draft)" notlarını kaldırmak | en son |
| 4 | Metindeki dosya yollarını (`src/...`, `results_v2/...`) bir "Reproducibility Map" tablosuna taşımak | en son |
| 5 | PDF'te ± işaretinin metin katmanında doğru çıkması (`\pdfgentounicode` ya da eşdeğeri; build tarafı, Hasan) | yapılacak |
| 6 | `paper/latex/arxiv/` paketinin arXiv'in derleyicisinde derlendiğini kontrol etmek (`main.tex`, `main.bbl`, 6 şekil mevcut) | yapılacak |
| 7 | Katkı beyanı, veri ve kod erişilebilirliği, çıkar çatışması beyanı | onay bekliyor |
| 8 | Kodun arXiv sürümüne karşılık gelen bir sürüm etiketi (git tag, isteğe bağlı Zenodo DOI) | en son |
