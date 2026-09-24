"""Omurgalar ve füzyon operatörleri — parametre bütçesi eşlenmiş.

Tasarım ilkesi: füzyon operatörleri yalnızca *nasıl birleştirdikleriyle* ayrışmalı,
kapasiteleriyle değil. Bu yüzden:

- Her iki-dallı model aynı CNN1D + CNN2D omurgasını kullanır.
- Her füzyon modülü aynı parametre bütçesine (`FUSION_BUDGET`) ayarlanır; gizli katman
  genişliği operatöre göre çözülür.
- Dropout, sınıflandırma başlığı ve gömme boyutu tüm modellerde aynıdır.
- Ayrıca `raw1d_wide` / `spec2d_wide` kontrolleri vardır: tek modaliteli ama füzyon
  modelleriyle *aynı toplam parametreye* sahip. Bu, "füzyon mu yardım etti yoksa
  fazladan parametre mi" sorusunu ayırır.

İlk (notebook tabanlı) sürümde Late 0.038M, Attention 0.054M, Gated 0.087M parametreye sahipti ve
Dropout yalnızca LateFusion'da vardı; bu haliyle füzyon operatörünün etkisi ölçülemiyordu.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSION_BUDGET = 40_000  # füzyon modülü başına hedef parametre sayısı


def _solve_hidden(kind: str, dim: int, ncls: int, budget: int) -> int:
    """Verilen bütçeye en yakın gizli katman genişliğini çözer."""
    if kind == "late":      # 2d -> h -> ncls
        h = (budget - ncls) / (2 * dim + 1 + ncls)
    elif kind == "gated":   # 2d -> h -> d (sigmoid), d -> ncls
        h = (budget - dim - (dim * ncls + ncls)) / (2 * dim + 1 + dim)
    elif kind == "attention":  # 2d -> h -> 2 (softmax), d -> ncls
        h = (budget - 2 - (dim * ncls + ncls)) / (2 * dim + 1 + 2)
    else:
        raise ValueError(kind)
    return max(8, int(round(h)))


class CNN1D(nn.Module):
    """in_ch: girdi kanal sayisi. Bonn tek kanalli (1), CHB-MIT 18 kanalli.
    Yalnizca ilk evrisim katmani degisir; in_ch=1 iken davranis oncekiyle ayni."""
    def __init__(self, dim: int = 128, width: int = 16, in_ch: int = 1):
        super().__init__()
        def blk(ci, co, k=7):
            return nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2),
                                 nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2))
        c1, c2, c3 = width, width * 2, width * 4
        self.net = nn.Sequential(blk(in_ch, c1), blk(c1, c2), blk(c2, c3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(c3, dim))

    def forward(self, x):
        return self.head(self.net(x))


class CNN2D(nn.Module):
    """in_ch: spektrogramin kanal sayisi. Bonn 1, CHB-MIT icin kanal basina
    bir spektrogram dusunuldugunde 18."""
    def __init__(self, dim: int = 128, width: int = 16, in_ch: int = 1):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1),
                                 nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2))
        c1, c2, c3 = width, width * 2, width * 4
        self.net = nn.Sequential(blk(in_ch, c1), blk(c1, c2), blk(c2, c3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, dim))

    def forward(self, x):
        return self.head(self.net(x))


# --- Tek modaliteli modeller -------------------------------------------------

class Unimodal1D(nn.Module):
    def __init__(self, dim=128, ncls=3, p_drop=0.3, width=16, in_ch=1):
        super().__init__()
        self.f = CNN1D(dim, width, in_ch)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(dim, ncls))

    def forward(self, x1d, x2d=None):
        return self.head(self.f(x1d))


class Unimodal2D(nn.Module):
    def __init__(self, dim=128, ncls=3, p_drop=0.3, width=16, in_ch=1):
        super().__init__()
        self.f = CNN2D(dim, width, in_ch)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(dim, ncls))

    def forward(self, x1d=None, x2d=None):
        return self.head(self.f(x2d))


# --- Füzyon operatörleri -----------------------------------------------------

class LateFusion(nn.Module):
    """Öznitelik düzeyinde birleştirme: concat -> MLP -> sınıf."""
    def __init__(self, dim=128, ncls=3, p_drop=0.3, budget=FUSION_BUDGET, in_ch=1):
        super().__init__()
        h = _solve_hidden("late", dim, ncls, budget)
        self.f1, self.f2 = CNN1D(dim, in_ch=in_ch), CNN2D(dim, in_ch=in_ch)
        self.fuse = nn.Sequential(nn.Dropout(p_drop), nn.Linear(2 * dim, h), nn.ReLU())
        self.cls = nn.Linear(h, ncls)

    def forward(self, x1d, x2d):
        z = torch.cat([self.f1(x1d), self.f2(x2d)], dim=1)
        return self.cls(self.fuse(z))


class GatedFusion(nn.Module):
    """Öğrenilebilir kapı: g = sigma(MLP([z1,z2])), z = g*z1 + (1-g)*z2."""
    def __init__(self, dim=128, ncls=3, p_drop=0.3, budget=FUSION_BUDGET, in_ch=1):
        super().__init__()
        h = _solve_hidden("gated", dim, ncls, budget)
        self.f1, self.f2 = CNN1D(dim, in_ch=in_ch), CNN2D(dim, in_ch=in_ch)
        self.gate = nn.Sequential(nn.Dropout(p_drop), nn.Linear(2 * dim, h), nn.ReLU(),
                                  nn.Linear(h, dim), nn.Sigmoid())
        self.cls = nn.Linear(dim, ncls)

    def forward(self, x1d, x2d):
        z1, z2 = self.f1(x1d), self.f2(x2d)
        g = self.gate(torch.cat([z1, z2], dim=1))
        return self.cls(g * z1 + (1 - g) * z2)

    def fusion_weights(self, x1d, x2d):
        z1, z2 = self.f1(x1d), self.f2(x2d)
        return self.gate(torch.cat([z1, z2], dim=1))


class AttentionFusion(nn.Module):
    """Modalite düzeyinde softmax dikkat: z = a1*z1 + a2*z2."""
    def __init__(self, dim=128, ncls=3, p_drop=0.3, budget=FUSION_BUDGET, in_ch=1):
        super().__init__()
        h = _solve_hidden("attention", dim, ncls, budget)
        self.f1, self.f2 = CNN1D(dim, in_ch=in_ch), CNN2D(dim, in_ch=in_ch)
        self.att = nn.Sequential(nn.Dropout(p_drop), nn.Linear(2 * dim, h), nn.ReLU(),
                                 nn.Linear(h, 2))
        self.cls = nn.Linear(dim, ncls)

    def forward(self, x1d, x2d):
        z1, z2 = self.f1(x1d), self.f2(x2d)
        a = torch.softmax(self.att(torch.cat([z1, z2], dim=1)), dim=1)
        return self.cls(a[:, 0:1] * z1 + a[:, 1:2] * z2)

    def fusion_weights(self, x1d, x2d):
        z1, z2 = self.f1(x1d), self.f2(x2d)
        return torch.softmax(self.att(torch.cat([z1, z2], dim=1)), dim=1)


class EarlyFusion(nn.Module):
    """Girdi düzeyine yakın birleştirme.

    Not: 1D sinyal ile 2D spektrogramı gerçek anlamda "ham düzeyde" birleştirmek mümkün
    değildir (farklı boyutlar). Literatürde 'early fusion' diye anılan bu kurgu aslında
    *intermediate* füzyondur: sinyal sığ bir kodlayıcıdan geçirilip frekans ekseninde
    yayılarak spektrograma ek kanal olarak eklenir. Makalede bu şekilde adlandırılmalıdır.
    """
    def __init__(self, dim=128, ncls=3, p_drop=0.3, wave_ch=8, width=26,
                 budget=FUSION_BUDGET, in_ch=1):
        super().__init__()
        # Sinyal, spektrogramın zaman çözünürlüğüne indirgenir. Aksi hâlde 4097 uzunluk
        # frekans ekseninde yayıldığında (B, 9, 59, 2048) boyutunda dev bir tensör oluşur;
        # bu hem yavaştır hem de spektrogramı bilinçsizce yukarı örneklemek demektir.
        self.wave_enc = nn.Sequential(
            nn.Conv1d(in_ch, 8, 7, stride=4, padding=3), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, 8, 5, stride=4, padding=2), nn.BatchNorm1d(8), nn.ReLU(),
            nn.Conv1d(8, wave_ch, 3, stride=2, padding=1), nn.BatchNorm1d(wave_ch), nn.ReLU(),
        )
        def blk(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1),
                                 nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2))
        c1, c2, c3 = width, width * 2, width * 4
        self.backbone = nn.Sequential(blk(in_ch + wave_ch, c1), blk(c1, c2), blk(c2, c3))
        self.proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c3, dim))
        h = _solve_hidden("late", dim // 2, ncls, budget // 2)
        self.cls = nn.Sequential(nn.Dropout(p_drop), nn.Linear(dim, h), nn.ReLU(),
                                 nn.Linear(h, ncls))

    def forward(self, x1d, x2d):
        B, _, Fdim, T = x2d.shape
        wf = self.wave_enc(x1d)                              # [B, wave_ch, T'']
        wf = F.adaptive_avg_pool1d(wf, T)                    # spektrogramın T'sine hizala
        wf2d = wf.unsqueeze(2).expand(B, wf.shape[1], Fdim, T)
        return self.cls(self.proj(self.backbone(torch.cat([x2d, wf2d], dim=1))))


# --- Fabrika -----------------------------------------------------------------

# Genişlik çarpanları, tek modaliteli "wide" kontrollerin toplam parametre sayısını
# iki-dallı füzyon modelleriyle eşitlemek için kalibre edilmiştir (bkz. report_params).
WIDE_WIDTH_1D = 34
WIDE_WIDTH_2D = 30
EARLY_WIDTH = 26


def build(name: str, ncls: int, dim: int = 128, p_drop: float = 0.3,
          fusion_budget: int | None = None, in_ch: int = 1) -> nn.Module:
    """`fusion_budget` verilirse füzyon modüllerinin parametre bütçesini değiştirir.

    Duyarlılık analizinde kullanılır: sonuç, seçilen bütçeye bağlı mı?
    """
    b = FUSION_BUDGET if fusion_budget is None else fusion_budget
    if name == "raw1d":
        return Unimodal1D(dim, ncls, p_drop, in_ch=in_ch)
    if name == "spec2d":
        return Unimodal2D(dim, ncls, p_drop, in_ch=in_ch)
    if name == "raw1d_wide":
        return Unimodal1D(dim, ncls, p_drop, width=WIDE_WIDTH_1D, in_ch=in_ch)
    if name == "spec2d_wide":
        return Unimodal2D(dim, ncls, p_drop, width=WIDE_WIDTH_2D, in_ch=in_ch)
    if name == "late":
        return LateFusion(dim, ncls, p_drop, budget=b, in_ch=in_ch)
    if name == "gated":
        return GatedFusion(dim, ncls, p_drop, budget=b, in_ch=in_ch)
    if name == "attention":
        return AttentionFusion(dim, ncls, p_drop, budget=b, in_ch=in_ch)
    if name == "early":
        return EarlyFusion(dim, ncls, p_drop, budget=b, in_ch=in_ch)
    raise ValueError(f"bilinmeyen model: {name}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def report_params(ncls: int = 3, dim: int = 128) -> dict[str, int]:
    names = ["raw1d", "spec2d", "raw1d_wide", "spec2d_wide",
             "late", "gated", "attention", "early"]
    out = {n: count_params(build(n, ncls, dim)) for n in names}
    # score fusion tek bir ağ değildir: iki tek-modaliteli modelin toplamı
    out["score"] = out["raw1d"] + out["spec2d"]
    return out


if __name__ == "__main__":
    for k, v in report_params().items():
        print(f"{k:14s} {v:>9,d}")
