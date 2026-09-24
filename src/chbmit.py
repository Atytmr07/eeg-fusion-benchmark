"""CHB-MIT Scalp EEG veri seti: EDF okuma, anotasyon ayrıştırma, pencereleme.

Neden kendi EDF okuyucumuz var: EDF biçimi (Kemp ve ark. 1992) sabit uzunluklu ASCII
başlık ve int16 veri kayıtlarından oluşur, okunması yüz satırdan kısadır. mne veya
pyedflib kurmak yerine okuyucuyu burada tutmak, bağımlılık yüzeyini küçültür ve
dosyanın nasıl yorumlandığını görünür kılar. Ölçek dönüşümü ve kanal seçimi bu
projenin sonuçlarını doğrudan etkilediği için görünür olması tercih edildi.

Veri kaynağı: https://physionet.org/content/chbmit/1.0.0/
Lisans: Open Data Commons Attribution License v1.0
Künye: Shoeb, A. (2009), MIT PhD thesis; Goldberger ve ark. (2000), Circulation 101(23).

Bonn'dan farkları (tasarımı etkileyenler):
  - 23 kanal (bazı kayıtlarda 24 veya 26), Bonn tek kanal
  - 256 Hz, Bonn 173.61 Hz
  - Sürekli kayıt, Bonn kesilmiş segmentler
  - Sınıf dengesizliği aşırı: nöbet toplam sürenin yüzde 1'inden azı
  - Denek kimliği VAR, dolayısıyla denek bazlı çapraz doğrulama mümkün.
    Bonn'da bu mümkün değildi; bu veri setinin asıl getirisi budur.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_ROOT

CHB_ROOT = DATA_ROOT / "chbmit"
FS_CHB = 256.0

# Standart bipolar uzunlamasına montaj, 18 kanal.
#
# Neden sabit bir liste, neden kesişim değil: CHB-MIT'te kanal kümesi yalnızca
# denekler arasında değil, aynı deneğin kayıtları arasında da değişir. chb12'nin bir
# bölümü CS2 referanslı bir montajla kaydedilmiştir (C3-CS2, CP2-CS2 gibi) ve bu
# kayıtlar standart montajla **tek bir kanalı bile paylaşmaz**. Bu yüzden tüm
# kayıtların kesişimi boş kümedir; naif kesişim sessizce sıfır kanal üretir.
#
# Sabit hedef listeyle 686 kaydın 683'ü ve 198 nöbetin 185'i korunur, 24 deneğin
# hepsi kullanılabilir kalır. Düşen 3 kayıt ve 13 nöbetin tamamı chb12'nin farklı
# montajla kaydedilmiş bölümündendir.
TARGET_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
)


# --- EDF okuma ---------------------------------------------------------------

@dataclass
class EdfHeader:
    n_records: int
    record_duration: float        # saniye
    labels: list[str]
    samples_per_record: list[int]
    phys_min: np.ndarray
    phys_max: np.ndarray
    dig_min: np.ndarray
    dig_max: np.ndarray
    header_bytes: int

    @property
    def fs(self) -> list[float]:
        return [s / self.record_duration for s in self.samples_per_record]

    @property
    def duration(self) -> float:
        return self.n_records * self.record_duration


def read_edf_header(path: Path) -> EdfHeader:
    """EDF sabit başlığını okur. Biçim: 256 bayt genel + kanal başına 256 bayt."""
    with open(path, "rb") as fh:
        raw = fh.read(256)
        if len(raw) < 256:
            raise ValueError(f"{path.name}: başlık kısa, dosya bozuk olabilir")
        header_bytes = int(raw[184:192].decode("ascii", "replace").strip() or 0)
        n_records = int(raw[236:244].decode("ascii", "replace").strip())
        record_duration = float(raw[244:252].decode("ascii", "replace").strip())
        ns = int(raw[252:256].decode("ascii", "replace").strip())

        def field(width: int) -> list[str]:
            buf = fh.read(width * ns).decode("ascii", "replace")
            return [buf[i * width:(i + 1) * width].strip() for i in range(ns)]

        labels = field(16)
        field(80)                                   # transducer, kullanılmıyor
        field(8)                                    # physical dimension
        phys_min = np.array([float(x) for x in field(8)])
        phys_max = np.array([float(x) for x in field(8)])
        dig_min = np.array([float(x) for x in field(8)])
        dig_max = np.array([float(x) for x in field(8)])
        field(80)                                   # prefiltering
        spr = [int(x) for x in field(8)]

    return EdfHeader(n_records, record_duration, labels, spr,
                     phys_min, phys_max, dig_min, dig_max,
                     header_bytes or (256 * (ns + 1)))


def read_edf(path: Path, channels: list[str] | None = None) -> tuple[np.ndarray, list[str], float]:
    """EDF sinyallerini fiziksel birimde (mikrovolt) döndürür.

    Dönen dizi (kanal, örnek). Tüm kanalların aynı örnekleme hızında olduğu
    varsayılır; CHB-MIT'te bu koşul sağlanır ve sağlanmazsa hata verilir.
    """
    h = read_edf_header(path)
    if len(set(h.samples_per_record)) != 1:
        raise ValueError(f"{path.name}: kanallar farklı örnekleme hızında, "
                         f"bu okuyucu bunu desteklemiyor")
    spr = h.samples_per_record[0]
    ns = len(h.labels)

    data = np.fromfile(path, dtype="<i2", offset=h.header_bytes)
    expected = h.n_records * ns * spr
    if data.size < expected:                        # kesik dosya: kayıt sayısını düşür
        h.n_records = data.size // (ns * spr)
        expected = h.n_records * ns * spr
    data = data[:expected].reshape(h.n_records, ns, spr)
    data = data.transpose(1, 0, 2).reshape(ns, -1).astype(np.float32)

    # EDF ölçek dönüşümü: fiziksel = (dijital - dig_min) * gain + phys_min
    span = np.where(h.dig_max - h.dig_min == 0, 1.0, h.dig_max - h.dig_min)
    gain = ((h.phys_max - h.phys_min) / span).astype(np.float32)
    offs = h.phys_min.astype(np.float32)
    dmin = h.dig_min.astype(np.float32)
    data = (data - dmin[:, None]) * gain[:, None] + offs[:, None]

    labels = h.labels
    if channels is not None:
        idx = [labels.index(c) for c in channels]
        data, labels = data[idx], list(channels)
    return data, labels, h.fs[0]


# --- Nöbet anotasyonları -----------------------------------------------------

@dataclass
class Seizure:
    file: str
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


def parse_summary(path: Path, strict: bool = False) -> dict[str, list[Seizure]]:
    """chbNN-summary.txt dosyasından dosya başına nöbet aralıklarını çıkarır.

    Ayrıştırma **sıra tabanlıdır**, nöbet numarasına güvenmez. Gerekçesi kaynak
    veride bulunan bir yazım hatasıdır: chb09_08.edf bloğunda ikinci nöbetin bitiş
    satırı "Seizure 1 End Time" diye etiketlenmiştir:

        Seizure 1 Start Time: 2951 seconds
        Seizure 1 End Time:   3030 seconds
        Seizure 2 Start Time: 9196 seconds
        Seizure 1 End Time:   9267 seconds     <- "Seizure 2" olmalıydı

    Numaraya göre eşleştiren bir ayrıştırıcı burada 2951-9267 arasını tek nöbet
    sayar, yani 6316 saniyelik var olmayan bir nöbet üretir ve yüzlerce nöbetsiz
    pencereyi ictal etiketler. Sıra tabanlı okuma bu hataya bağışıktır: her
    "Start Time" yeni bir nöbet açar, her "End Time" en son açılanı kapatır.

    Her blokta bulunan nöbet sayısı "Number of Seizures in File" ile karşılaştırılır.
    strict=True ise uyuşmazlıkta hata verilir, aksi halde uyarı basılır.

    Anotasyonların ikili `.edf.seizures` dosyaları da vardır, ancak biçimi
    belgelenmemiştir; özet dosyası insan tarafından okunabilir ve resmi kaynaktır.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, list[Seizure]] = {}
    current: str | None = None
    declared: dict[str, int] = {}
    open_start: float | None = None

    def close_block() -> None:
        if current is None:
            return
        want, got = declared.get(current), len(out.get(current, []))
        if want is not None and want != got:
            msg = (f"{path.name}: {current} için beyan edilen nöbet sayısı {want}, "
                   f"ayrıştırılan {got}")
            if strict:
                raise ValueError(msg)
            print(f"uyarı: {msg}")

    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            close_block()
            current = m.group(1)
            out.setdefault(current, [])
            open_start = None
            continue
        if current is None:
            continue
        m = re.match(r"Number of Seizures in File:\s*(\d+)", line)
        if m:
            declared[current] = int(m.group(1))
            continue
        m = re.match(r"Seizure.*Start Time:\s*(\d+)\s*seconds", line)
        if m:
            if open_start is not None:
                print(f"uyarı: {path.name}: {current} içinde kapatılmamış nöbet "
                      f"(başlangıç {open_start}) atlandı")
            open_start = float(m.group(1))
            continue
        m = re.match(r"Seizure.*End Time:\s*(\d+)\s*seconds", line)
        if m:
            end = float(m.group(1))
            if open_start is None:
                print(f"uyarı: {path.name}: {current} içinde başlangıcı olmayan "
                      f"bitiş satırı ({end}) atlandı")
                continue
            out[current].append(Seizure(current, open_start, end))
            open_start = None
    close_block()
    return out


# --- Kanal uyumu -------------------------------------------------------------

def common_channels(files: list[Path]) -> list[str]:
    """Tüm kayıtlarda ortak olan kanalları, ilk dosyadaki sırayla döndürür.

    CHB-MIT'te kanal kümesi kayıtlar arasında değişebilir; bazı dosyalarda yinelenen
    veya boş ('-') kanallar bulunur. Modelin her kayıtta aynı girdiyi görmesi için
    kesişim alınır.
    """
    sets, order = [], []
    for f in files:
        labs = read_edf_header(f).labels
        seen, uniq = set(), []
        for l in labs:
            if l in ("-", "") or l in seen:
                continue
            seen.add(l)
            uniq.append(l)
        sets.append(seen)
        if not order:
            order = uniq
    keep = set.intersection(*sets) if sets else set()
    return [l for l in order if l in keep]


def subject_files(subject: str, require_channels: bool = True) -> list[Path]:
    """Deneğin EDF dosyaları.

    Kalıp `{subject}*.edf`, `{subject}_*.edf` değil: chb17'nin kayıtları
    `chb17a_03.edf`, `chb17b_69.edf`, `chb17c_...` biçiminde adlandırılmıştır ve
    alt çizgi bekleyen bir kalıp bu deneğin tamamını sessizce atlar.

    require_channels=True ise TARGET_CHANNELS'ın tamamını içermeyen kayıtlar elenir.
    """
    files = sorted((CHB_ROOT / subject).glob(f"{subject}*.edf"))
    if not require_channels:
        return files
    want = set(TARGET_CHANNELS)
    return [f for f in files if want <= set(read_edf_header(f).labels)]


# --- Pencereleme -------------------------------------------------------------

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def seizure_ids(subject: str) -> dict[str, list[tuple[float, float, str]]]:
    """Deneğin her nöbetine kalıcı bir kimlik verir: chb09_08.edf#2 gibi.

    Kimlik, pencereleri nöbet düzeyinde gruplamak için gerekir. Örtüşmeli
    pencerelemede aynı nöbetin pencereleri neredeyse aynıdır; bir kısmı eğitimde
    bir kısmı testte kalırsa model ezberlediğini genelleme sanır. Bu hata sınıfı
    CHB-MIT literatüründe belgelenmiştir (Ali ve ark. 2024, doi:10.1098/rsos.230601).
    """
    summ = parse_summary(CHB_ROOT / subject / f"{subject}-summary.txt")
    out: dict[str, list[tuple[float, float, str]]] = {}
    for fname, zs in summ.items():
        for k, z in enumerate(zs, 1):
            out.setdefault(fname, []).append((z.start_s, z.end_s, f"{fname}#{k}"))
    return out


def plan_windows(path: Path, seizures: list[Seizure], win_s: float = 10.0,
                 stride_s: float | None = None,
                 guard_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Pencere etiketlerini **sinyali okumadan** hesaplar.

    Etiket yalnızca pencere zamanına ve nöbet aralıklarına bağlıdır; kayıt süresi
    EDF başlığından okunur. Bu sayede hangi pencerelerin tutulacağına karar vermek
    için 42 dosyanın 1.7 GB verisini belleğe almak gerekmez. 22 denekte aynı yaklaşım
    olmadan bellek yetmez.

    Dönenler: y (0 non-ictal / 1 ictal), t0 (pencere başlangıcı, saniye).

    guard_s > 0 ise nöbetle kısmen kesişen sınır pencereleri atılır. Sınır
    pencerelerinin etiketi belirsizdir; dahil edilip edilmemesi bir tasarım
    tercihidir ve duyarlılık kontrolü olarak iki türlü de koşulur.
    """
    h = read_edf_header(path)
    fs = int(round(h.fs[0]))
    n = int(win_s * fs)
    step = int((stride_s if stride_s else win_s) * fs)
    total = int(h.duration * fs)
    sz = [(s.start_s, s.end_s) for s in seizures]

    y, t0 = [], []
    for start in range(0, total - n + 1, step):
        a0, a1 = start / fs, (start + n) / fs
        ov = max((_overlap(a0, a1, b0, b1) for b0, b1 in sz), default=0.0)
        if ov == 0.0:
            lab = 0
        elif ov >= win_s - 1e-9:                    # pencere tamamen nöbet içinde
            lab = 1
        else:                                       # sınır penceresi
            if guard_s > 0:
                continue
            lab = 1
        y.append(lab)
        t0.append(a0)
    return np.array(y, np.int64), np.array(t0, np.float32)


def extract_windows(path: Path, channels: list[str], starts_s: np.ndarray,
                    win_s: float = 10.0) -> np.ndarray:
    """Verilen başlangıç zamanlarındaki pencereleri okur. Dönen: (pencere, kanal, örnek)."""
    x, _, fs = read_edf(path, channels)
    fs = int(round(fs))
    n = int(win_s * fs)
    out = np.empty((len(starts_s), len(channels), n), np.float32)
    for i, t in enumerate(starts_s):
        s = int(round(float(t) * fs))
        out[i] = x[:, s:s + n]
    return out


def window_file(path: Path, seizures: list[Seizure], channels: list[str],
                win_s: float = 10.0, stride_s: float | None = None,
                guard_s: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bir kaydı pencereler ve sinyali de döndürür. Tek dosyalık kontroller içindir;
    denek düzeyinde build_subject kullanılmalı, o belleğe tüm kaydı almaz."""
    y, t0 = plan_windows(path, seizures, win_s, stride_s, guard_s)
    if len(y) == 0:
        fs = int(round(read_edf_header(path).fs[0]))
        return (np.empty((0, len(channels), int(win_s * fs)), np.float32),
                y, t0)
    return extract_windows(path, channels, t0, win_s), y, t0


def build_subject(subject: str, win_s: float = 10.0, stride_s: float | None = None,
                  neg_per_pos: float = 4.0, seed: int = 20260727,
                  guard_s: float = 0.0, verbose: bool = True):
    """Bir deneğin tüm kayıtlarını pencereler ve non-ictal sınıfı alt örnekler.

    neg_per_pos: her ictal pencereye karşılık tutulacak non-ictal pencere sayısı.
    Bu oran ölçülen performansı doğrudan belirler, bu yüzden döndürülen meta
    bilgisine yazılır ve raporlanır. Alt örnekleme tohumludur.

    Not: alt örnekleme yalnızca *eğitim ve değerlendirme kümesinin boyutunu*
    küçültmek içindir. Gerçek dağılımda nöbet binde birler mertebesindedir; bu
    nedenle buradan çıkan mutlak sayılar klinik yanlış alarm oranını temsil etmez.
    Saatte yanlış alarm metriği ayrıca, alt örneklenmemiş zaman ekseni üzerinden
    hesaplanmalıdır.
    """
    all_files = subject_files(subject, require_channels=False)
    files = subject_files(subject)
    if not files:
        raise FileNotFoundError(
            f"{subject}: hedef montajı içeren EDF yok ({CHB_ROOT / subject}, "
            f"{len(all_files)} kayıt tarandı)")
    summ = parse_summary(CHB_ROOT / subject / f"{subject}-summary.txt")
    chans = list(TARGET_CHANNELS)
    n_skipped = len(all_files) - len(files)
    sz_skipped = sum(len(summ.get(f.name, []))
                     for f in set(all_files) - set(files))

    # 1. geçiş: yalnızca başlıklardan etiket planı. Sinyal okunmaz.
    ys, fnames, t0s = [], [], []
    for f in files:
        y_f, t_f = plan_windows(f, summ.get(f.name, []), win_s, stride_s, guard_s)
        if len(y_f) == 0:
            continue
        ys.append(y_f)
        t0s.append(t_f)
        fnames += [f.name] * len(y_f)
    y = np.concatenate(ys)
    t0_arr = np.concatenate(t0s)
    files_arr = np.array(fnames)

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep_neg = neg
    if neg_per_pos > 0 and len(neg) > neg_per_pos * len(pos):
        rng = np.random.default_rng(seed)
        keep_neg = rng.choice(neg, int(neg_per_pos * len(pos)), replace=False)
    idx = np.sort(np.concatenate([pos, keep_neg]))

    # 2. geçiş: yalnızca seçilen pencereler okunur, dosya başına tek okuma.
    by_name = {f.name: f for f in files}
    X = np.empty((len(idx), len(chans), int(win_s * FS_CHB)), np.float32)
    pos_in_out = {}
    for j, i in enumerate(idx):
        pos_in_out.setdefault(files_arr[i], []).append((j, t0_arr[i]))
    for name, items in pos_in_out.items():
        js = np.array([a for a, _ in items])
        ts = np.array([b for _, b in items], np.float32)
        X[js] = extract_windows(by_name[name], chans, ts, win_s)

    info = {
        "subject": subject, "n_files": len(files),
        "n_files_skipped_montage": n_skipped,
        "n_seizures_skipped_montage": sz_skipped,
        "channels": chans,
        "n_channels": len(chans), "fs": FS_CHB, "win_s": win_s,
        "stride_s": stride_s or win_s, "guard_s": guard_s,
        "n_windows_total": int(len(y)), "n_ictal": int(len(pos)),
        "n_nonictal_total": int(len(neg)), "n_nonictal_kept": int(len(keep_neg)),
        "neg_per_pos": neg_per_pos, "seed": seed,
        "ictal_fraction_raw": float(len(pos) / len(y)),
        # Saatte yanlış alarm metriği alt örneklenmemiş zaman ekseninden
        # hesaplanmalı; toplam kayıt süresi bu yüzden saklanır.
        "total_recording_hours": float(len(y) * (stride_s or win_s) / 3600.0),
    }
    if verbose:
        skip = (f", montaj uyuşmadığı için atlanan {n_skipped} kayıt / "
                f"{sz_skipped} nöbet") if n_skipped else ""
        print(f"{subject}: {len(files)} kayıt, {len(chans)} kanal{skip}, "
              f"{len(y)} pencere ({len(pos)} ictal, oran {len(pos)/len(y):.4%})")
        print(f"  alt örnekleme sonrası: {len(idx)} pencere "
              f"({len(pos)} ictal / {len(keep_neg)} non-ictal)")
    return X, y[idx], files_arr[idx], t0_arr[idx], info
