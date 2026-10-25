#!/usr/bin/env bash
# Build and push the three Kaggle GPU sessions (REVISION_PLAN.md section 2).
# NvidiaTeslaP100 is the default: Kaggle starts it at once (it assigned a single T4); T4 x2 queues for hours.
# Usage: revision/build.sh smoke|full [a..h]   (GPU=false for CPU-only, e.g. d)
# Each kernel = SMOKE flag + hsi_common.py + its run scripts, as one Kaggle script.
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:?smoke or full}; shift
smoke=$([ "$mode" = smoke ] && echo True || echo False)
owner=${KAGGLE_OWNER:-devgurucodes}
shape=$([ "${GPU:-true}" = true ] && echo " \"machine_shape\": \"${MACHINE:-NvidiaTeslaP100}\"," || true)
declare -A runs=([a]="r1_fewshot r6_timing" [b]="r3_transfer_local_to_ip r5_linear_probe" [c]="r4_transfer_ip_to_local r2_local" [d]="r7_shap" [e]="r8_disjoint r8_ip" [f]="r8_disjoint r8_paviau" [g]="r8_disjoint r8_salinas" [h]="r11_local_samples" [fc]="r8_disjoint r8_paviau_cpu" [gc]="r8_disjoint r8_salinas_cpu" [l]="r8_disjoint r12_lightgbm" [i]="r11b_masked" [j]="r13_moisture" [k]="r8_disjoint r18_classic" [m]="r8_disjoint r8_paviau_sfpad" [n]="r8_disjoint r17_maps r15_long r15_ip_a" [o]="r8_disjoint r15_long r15_ip_b" [p]="r8_disjoint r15_long r15_paviau" [s]="r8_disjoint r15_long r15_salinas_a" [t]="r8_disjoint r15_long r15_salinas_b" [q]="r13b_setup r13_moisture" [u]="r8_disjoint r15_long r15_sf_cpu_a" [v]="r8_disjoint r15_long r15_sf_cpu_b")

for s in "${@:-a b c}"; do
  dir=build/session-$s; mkdir -p "$dir"
  { echo "SMOKE = $smoke"; cat hsi_common.py
    for r in ${runs[$s]}; do echo; echo "# ===== $r"; grep -vE '^from (hsi_common|r8_disjoint|r15_long) import' "$r.py"; done
  } > "$dir/session.py"
  cat > "$dir/kernel-metadata.json" <<EOF
{"id": "$owner/rsl-session-$s", "title": "rsl-session-$s", "code_file": "session.py",
 "language": "python", "kernel_type": "script", "is_private": true,
 "enable_gpu": ${GPU:-true}, "enable_internet": true,$shape
 "dataset_sources": ["dev123123456/local-soil-hyperspectral-dataset"],
 "competition_sources": [], "kernel_sources": []}
EOF
  python -m kaggle kernels push -p "$dir"
done
