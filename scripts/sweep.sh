#!/bin/bash

# ============================================================
#  Bird's Eye View Parameter Sweep
#  Runs the compiled birdseye executable across a grid of
#  altitude and pitch values, saving each result with a
#  timestamped filename (handled by the executable itself).
#
#  Usage:
#    chmod +x sweep.sh
#    ./sweep.sh
#
#  Tweak the arrays below to focus on any region of interest.
# ============================================================

EXECUTABLE="../cv/build/birdseye"

# --- Parameters to sweep ---
ALTITUDES=(200 300 400)
PITCHES=(-60 -65 -70 -75 -80 -85)
ROLL=0
YAW=0
HFOV=110
VFOV=80

# ============================================================

if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: executable '$EXECUTABLE' not found."
    echo "Build first: mkdir build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

total=$(( ${#ALTITUDES[@]} * ${#PITCHES[@]} ))
count=0

echo "============================================"
echo "  Sweep: ${#ALTITUDES[@]} altitudes x ${#PITCHES[@]} pitches = $total runs"
echo "  HFOV=${HFOV}  VFOV=${VFOV}  ROLL=${ROLL}  YAW=${YAW}"
echo "============================================"

for ALT in "${ALTITUDES[@]}"; do
    for PITCH in "${PITCHES[@]}"; do
        count=$(( count + 1 ))
        echo ""
        echo "[$count/$total]  alt=${ALT}m  pitch=${PITCH}deg"
        $EXECUTABLE "$ALT" "$PITCH" "$ROLL" "$YAW" "$HFOV" "$VFOV"
    done
done

echo ""
echo "============================================"
echo "  Done. $total images saved to OUTPUT_PATH."
echo "============================================"