#!/usr/bin/env bash
# Tüm deney koşularını PARALEL çalıştırır.
# 14 çekirdek; koşu başına 2 torch thread -> 6 koşu x 2 = 12, sistem için pay bırakır.
set -u
PY=${PY:-python}
# Thread sayisi sonuclari degistirir ve artik kosu hash'ine dahildir (bkz.
# docs/10_OLASILIK_VE_TEKRARLANABILIRLIK.md). Degistirirseniz eski kosularla
# karsilastirilamaz.
THREADS=${THREADS:-2}
cd "$(dirname "$0")"
mkdir -p logs

launch () {
  local task=$1 norm=$2 tag=$3
  OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    "$PY" -m src.run_benchmark --task "$task" --norm "$norm" --threads "$THREADS" \
    > "logs/$tag.log" 2>&1 &
  echo "started $tag (pid $!)"
}

# --- Öncelik 1: makalenin taşıyıcı koşuları ------------------------------------
launch T1_3class              N2_z_zspec       main_T1_N2   # ana kıyaslama
launch T2_intracranial_binary N2_z_zspec       main_T2_N2   # elektrot karıştırıcısı yok
launch T1_3class              N1_z_rawspec     abl_T1_N1    # normalizasyon karşıtlığı
launch T3_classic_binary      N2_z_zspec       ref_T3_N2    # klasik ikili kurgu (referans)

# --- Öncelik 2: tamamlayıcı ablasyon kolları -----------------------------------
# Yukarıdakiler bittikten sonra çalıştırın; CPU'yu paylaşırlarsa hepsi yavaşlar.
# launch T1_3class            N0_raw_raw       abl_T1_N0
# launch T1_3class            N3_z_zspec_specz abl_T1_N3

echo "=== $(date '+%H:%M:%S') 6 kosu paralel basladi, bekleniyor ==="
wait
echo "=== $(date '+%H:%M:%S') HEPSI BITTI ==="
