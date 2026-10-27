#!/usr/bin/env bash
# Build the small, hash-locked checkpoint bundle needed by the visual server.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FILES=(
  deploy/jaka_visual_teacher_manifest.json
  artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j0_s1_probe500k/checkpoints/ppo_teacher_final.pt
  artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j1_place_006_008_1024_no_rtx/checkpoints/ppo_teacher_final.pt
  artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/teachers/jaka_static_sphere_moving_bowl_m0_final_geometry_smoke16/checkpoints/ppo_teacher_final.pt
)

for path in "${FILES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing bundle input: $path" >&2
    exit 2
  fi
done

EXPECTED=(
  14b1d8af7ecaa46e4a77c48434ca07d43e7d2f3e73672832af663ece2485ea7b
  a945612cfb694adc424b7264635c20a34279e89e08c9aa20dc93c79c47450759
  d522d818dccdb7a073708b75ccb44fb7bc9b98a5780e10f057b5faba3b01555b
)
for index in 0 1 2; do
  path="${FILES[$((index + 1))]}"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "${EXPECTED[$index]}" ]]; then
    echo "SHA-256 mismatch for $path" >&2
    exit 2
  fi
done

OUT_DIR="$REPO_ROOT/artifacts/server_transfer"
OUT="${JAKA_VISUAL_WEIGHT_BUNDLE:-$OUT_DIR/jaka_visual_teacher_payload_20261027.tar.gz}"
mkdir -p "$OUT_DIR"
tar -czf "$OUT" "${FILES[@]}"
sha256sum "$OUT"
echo "JAKA_VISUAL_WEIGHT_BUNDLE=$OUT"
