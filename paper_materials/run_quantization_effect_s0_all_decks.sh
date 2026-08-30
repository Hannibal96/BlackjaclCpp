#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
quantization_effect="$repo_root/build/bin/QuantizationEffect"

if [[ ! -x "$quantization_effect" ]]; then
    echo "QuantizationEffect is not built: $quantization_effect" >&2
    echo "Build it first with: cmake --build '$repo_root/build' --target QuantizationEffect" >&2
    exit 1
fi

checkpoints=(
    "paper_materials/blackjack results 5.2/alternating-checkpoints/h17c03_5b_qk_s0_decks=1_ss17=False_das=True_surr=no_peek=True"
    "paper_materials/blackjack results 5.2/alternating-checkpoints/h17c03_5b_qk_s0_decks=2_ss17=False_das=True_surr=no_peek=True"
    "paper_materials/blackjack results 5.2/alternating-checkpoints/h17c03_5b_qk_s0_decks=6_ss17=False_das=True_surr=no_peek=True"
    "paper_materials/blackjack 6:5 5.3/alternating-checkpoints/h17c03_5b_bj6-5_qk_s0_decks=1_ss17=False_das=True_surr=no_peek=True"
    "paper_materials/blackjack 6:5 5.3/alternating-checkpoints/h17c03_5b_bj6-5_qk_s0_decks=2_ss17=False_das=True_surr=no_peek=True"
    "paper_materials/blackjack 6:5 5.3/alternating-checkpoints/h17c03_5b_bj6-5_qk_s0_decks=6_ss17=False_das=True_surr=no_peek=True"
)

cd -- "$repo_root"

for checkpoint in "${checkpoints[@]}"; do
    printf '\n===== QuantizationEffect: %s =====\n\n' "$checkpoint"

    "$quantization_effect" \
        --checkpoint "$checkpoint" \
        --seed 12345 \
        --num-threads 10 \
        --kelly-measurements 1000 \
        --eval-rounds 5000000000 \
        --kelly-min 0.85 \
        --kelly-max 1.15
done
