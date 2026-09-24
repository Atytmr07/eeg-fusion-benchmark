"""Deney konfigürasyonu ve içerik-tabanlı çıktı dizini isimlendirmesi.

Her koşunun çıktısı, konfigürasyonun hash'iyle isimlendirilmiş bir dizine yazılır.
Böylece farklı ayarlarla üretilmiş sonuçlar birbirine karışamaz. Projenin ilk
(notebook tabanlı) sürümünde SEED=42 ve SEED=1337 koşularının aynı klasöre yazması
bu yüzden sorun oluyordu.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results_v2"

# Bonn veri seti sabitleri (Andrzejak ve ark. 2001).
# Kayıt sırasında 0.53-40 Hz band-pass uygulanmıştır; segmentler 23.6 s / 4097 örnek.
FS = 173.61
SEGMENT_LEN = 4097
SET_NAMES = ("A", "B", "C", "D", "E")

# A/B yüzey (scalp) EEG; C/D/E intrakraniyal derinlik elektrodu.
RECORDING_TYPE = {"A": "scalp", "B": "scalp", "C": "intracranial",
                  "D": "intracranial", "E": "intracranial"}

# --- Görev tanımları ---------------------------------------------------------
# T1: 3 sınıf. Zorluk Normal<->Interictal ayrımında.
# T2: intrakraniyal ikili. Elektrot tipi karıştırıcısından arınmış.
# T3: literatürdeki klasik ikili kurgu. Yalnızca referans olarak tutulur; log-varyans
#     baseline'ı bunu tek öznitelikle çözdüğü için ana deney değildir.
TASKS = {
    "T1_3class": {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2},
    "T2_intracranial_binary": {"C": 0, "D": 0, "E": 1},
    "T3_classic_binary": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 1},
}
CLASS_NAMES = {
    "T1_3class": ["Normal", "Interictal", "Ictal"],
    "T2_intracranial_binary": ["Interictal", "Ictal"],
    "T3_classic_binary": ["Non-seizure", "Seizure"],
}

# --- Normalizasyon ablasyonu -------------------------------------------------
# İlk (notebook tabanlı) sürümün fiilen çalıştırdığı kol N1'dir: 1D dal z-score'lu, 2D dal
# normalize edilmemiş sinyalden üretilmiş spektrogram görüyordu (cache hatası).
NORM_MODES = {
    "N0_raw_raw":       {"sig_z": False, "spec_from_z": False, "spec_z": False},
    "N1_z_rawspec":     {"sig_z": True,  "spec_from_z": False, "spec_z": False},
    "N2_z_zspec":       {"sig_z": True,  "spec_from_z": True,  "spec_z": False},
    "N3_z_zspec_specz": {"sig_z": True,  "spec_from_z": True,  "spec_z": True},
}

MODELS = ("raw1d", "spec2d", "early", "late", "gated", "attention", "score")


@dataclass(frozen=True)
class Config:
    task: str = "T1_3class"
    norm_mode: str = "N2_z_zspec"

    # Ön işleme. Bonn zaten 0.53-40 Hz band-limitli olduğu için ek filtre
    # varsayılan olarak kapalıdır; ablasyon için açılabilir.
    apply_lowpass: bool = False
    lowpass_hz: float = 40.0
    lowpass_order: int = 4

    # STFT
    nfft: int = 256
    win: int = 256
    hop: int = 128
    fmax: float = 40.0

    # Eğitim
    embed_dim: int = 128
    dropout: float = 0.3
    batch_size: int = 32
    epochs: int = 60
    # Erken durdurma, model henüz önemsiz çözümdeyken (tüm örneklere tek sınıf)
    # tetiklenebiliyordu: doğrulama macro-F1'i düz kaldığı için sayaç dolup eğitim
    # ~11. epoch'ta kesiliyor ve çökmüş ağırlıklar "en iyi" olarak saklanıyordu.
    # AUC 0.96 iken F1 0.40 olan koşular bu yüzden oluşuyordu. min_epochs bunu önler.
    min_epochs: int = 20
    patience: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    class_weighted_loss: bool = True

    # Değerlendirme
    n_folds: int = 5
    n_repeats: int = 5
    val_ratio: float = 0.2
    base_seed: int = 20260727

    # Çalışma zamanı thread sayısı. Deney tasarımının parçası gibi görünmese de
    # sonuçları değiştirdiği ölçüldü: kayan nokta toplama sırası thread sayısıyla
    # değişiyor ve fold başına makro F1 0.09'a kadar, model ortalaması 0.01'e kadar
    # kayıyor (bkz. docs/10_OLASILIK_VE_TEKRARLANABILIRLIK.md). Bu büyüklük
    # karşılaştırılan operatörler arası farklarla aynı mertebede olduğu için
    # koşu kimliğine dahil edilir.
    threads: int = 2

    models: tuple = field(default=MODELS)

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    def outdir(self) -> Path:
        """Koşunun çıktı dizini. Yan etkisizdir, dizini açmaz."""
        return RESULTS_ROOT / f"{self.task}__{self.norm_mode}__{self.hash()}"

    def save(self, outdir: Path | None = None) -> Path:
        d = Path(outdir) if outdir else self.outdir()
        d.mkdir(parents=True, exist_ok=True)
        p = d / "config.json"
        p.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        (d / "env.json").write_text(json.dumps(runtime_env(), indent=2), encoding="utf-8")
        return p


def runtime_env() -> dict:
    """Sonuçları etkileyebilecek ortam bilgisi. Hash'e girmez, yanına yazılır.

    threads alanı config'te tutulur; burada fiilen yürürlükte olan değer kaydedilir,
    çünkü OMP_NUM_THREADS süreç başlamadan ayarlanmazsa istenen ile gerçekleşen
    ayrışabilir.
    """
    import platform
    import sys

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    for mod in ("torch", "numpy", "scipy", "sklearn"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            env[mod] = None
    try:
        import torch
        env["torch_num_threads_effective"] = torch.get_num_threads()
        env["torch_num_interop_threads"] = torch.get_num_interop_threads()
    except ImportError:
        pass
    return env


def fold_seed(base_seed: int, repeat: int, fold: int, model: str) -> int:
    """Her (tekrar, fold, model) üçlüsü için deterministik ve bağımsız seed.

    Notebook'ta seed yalnızca en başta bir kez set ediliyordu; her model kurulumu
    ilerleyen global RNG akışından çekiyordu. Hücre sırası değişince sonuçlar
    değişiyordu. Bu fonksiyon o bağımlılığı ortadan kaldırır.
    """
    key = f"{base_seed}|{repeat}|{fold}|{model}".encode()
    return int(hashlib.sha1(key).hexdigest()[:8], 16) % (2**31 - 1)
