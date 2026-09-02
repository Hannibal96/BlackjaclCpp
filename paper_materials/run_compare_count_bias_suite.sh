#!/usr/bin/env bash

set -Eeuo pipefail

# Compare Hi-Lo policies under three betting-signal bias configurations while
# using the continuous (unrounded) true count for betting:
#
#   1. fixed_0p5pct: a fixed -0.5% neutral-count edge (bias = -0.005)
#   2. wizard_edge:  the rule/deck-specific Wizard-of-Odds edge
#   3. alternating:  the W3 quadratic-Kelly zero crossing, rescaled to the
#                    common Hi-Lo factor so this remains a bias-only test
#
# Full deviations are trained once per deck and reused across all three bias
# evaluations. Bias and continuous_betting_count affect wager sizing, not the
# integer count state used to train the playing-policy table. Reusing the same
# table therefore isolates the bias instead of adding independent RL noise.

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly COMPARE_BIN="${PROJECT_DIR}/scripts/compare_count_strategies.py"

readonly NUM_THREADS="${NUM_THREADS:-10}"
readonly NICE_LEVEL="${NICE_LEVEL:-10}"
readonly SEED="${SEED:-12345}"

readonly NUM_ROUNDS=5000000000
readonly EVAL_ROUNDS=5000000000
readonly KELLY_ROUNDS=1000000
readonly KELLY_MEASUREMENTS=1000
readonly KELLY_MIN=0.50
readonly KELLY_MAX=1.00
readonly KELLY_STEP=0.05
readonly SAMPLE_ROUNDS=100000000
readonly DIFF_THRESHOLD=0.005

# Keep the Hi-Lo slope fixed. The Kelly multiplier search absorbs its overall
# scale, while changing bias/factor changes the signal's zero crossing.
readonly HILO_FACTOR=0.005
readonly FIXED_0P5_PERCENT_BIAS=-0.005

readonly SUITE_NAME="${SUITE_NAME:-h17c03_5b_hilo_bias_continuous}"
readonly SUITE_ROOT="${PROJECT_DIR}/checkpoints/performance-suites/${SUITE_NAME}"
readonly LOG_DIR="${SUITE_ROOT}/logs"

# Neutral-count player edges in decimal units. These are the negatives of the
# cut-card house edges in json_wod_edge.json for the suite's game rules.
declare -A WIZARD_GAME_BIAS=(
  ["1.5:1"]="-0.0015945"
  ["1.5:2"]="-0.0045688"
  ["1.5:6"]="-0.0063873"
  ["1.2:1"]="-0.0155422"
  ["1.2:2"]="-0.0183038"
  ["1.2:6"]="-0.0199842"
)

TRAINING_FLAGS=(
  --num-rounds "${NUM_ROUNDS}"
  --stop-mode diff
  --sample-rounds "${SAMPLE_ROUNDS}"
  --diff-threshold "${DIFF_THRESHOLD}"
)

EVALUATION_FLAGS=(
  --eval-rounds "${EVAL_ROUNDS}"
  --kelly-rounds "${KELLY_ROUNDS}"
  --kelly-measurements "${KELLY_MEASUREMENTS}"
  --kelly-min "${KELLY_MIN}"
  --kelly-max "${KELLY_MAX}"
  --kelly-step "${KELLY_STEP}"
  --num-threads "${NUM_THREADS}"
  --seed "${SEED}"
)

require_executable() {
  local path="$1"
  if [[ ! -x "${path}" ]]; then
    printf 'Required executable not found: %s\n' "${path}" >&2
    exit 1
  fi
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "${command_name}" >&2
    exit 1
  fi
}

wizard_bias_for() {
  local blackjack_pay="$1"
  local deck="$2"
  local payout_key
  local key

  printf -v payout_key '%.1f' "${blackjack_pay}"
  key="${payout_key}:${deck}"
  if [[ -z "${WIZARD_GAME_BIAS[${key}]+configured}" ]]; then
    printf 'No Wizard-of-Odds bias for payout=%s, decks=%s\n' \
      "${blackjack_pay}" "${deck}" >&2
    return 1
  fi
  printf '%s\n' "${WIZARD_GAME_BIAS[${key}]}"
}

alternating_w3_path_for() {
  local blackjack_pay="$1"
  local deck="$2"
  local alternating_root
  local alternating_prefix

  case "${blackjack_pay}" in
    1.5)
      alternating_root="${PROJECT_DIR}/paper_materials/blackjack results 5.2/alternating-checkpoints"
      alternating_prefix="h17c03_5b_qk_s0"
      ;;
    1.2)
      alternating_root="${PROJECT_DIR}/paper_materials/blackjack 6:5 5.3/alternating-checkpoints"
      alternating_prefix="h17c03_5b_bj6-5_qk_s0"
      ;;
    *)
      printf 'Unsupported blackjack payout: %s\n' "${blackjack_pay}" >&2
      return 1
      ;;
  esac

  printf '%s/%s_decks=%s_ss17=False_das=True_surr=no_peek=True/W3.json\n' \
    "${alternating_root}" "${alternating_prefix}" "${deck}"
}

alternating_bias_for() {
  local blackjack_pay="$1"
  local deck="$2"
  local artifact
  local learned_factor
  local learned_bias

  artifact="$(alternating_w3_path_for "${blackjack_pay}" "${deck}")"
  if [[ ! -f "${artifact}" ]]; then
    printf 'Missing alternating W3 artifact: %s\n' "${artifact}" >&2
    return 1
  fi

  learned_factor="$(jq -er '.count_config.factor' "${artifact}")"
  learned_bias="$(jq -er '.count_config.bias' "${artifact}")"

  # Preserve the learned zero crossing -bias/factor while holding the Hi-Lo
  # factor at 0.005: b_test = b_W3 * (0.005 / a_W3).
  awk -v bias="${learned_bias}" \
      -v factor="${learned_factor}" \
      -v target_factor="${HILO_FACTOR}" \
      'BEGIN {
         if (factor == 0) exit 1
         printf "%.17g\n", bias * target_factor / factor
       }'
}

bias_threshold() {
  local bias="$1"
  awk -v bias="${bias}" -v factor="${HILO_FACTOR}" \
    'BEGIN { printf "%.9f\n", -bias / factor }'
}

run_case() {
  local label="$1"
  shift

  local done_file="${LOG_DIR}/${label}.done"
  local log_file="${LOG_DIR}/${label}.console.log"

  if [[ -f "${done_file}" ]]; then
    printf '\n[%s] Skipping completed case: %s\n' \
      "$(date -u +%FT%TZ)" "${label}"
    return 0
  fi

  printf '\n[%s] Starting case: %s\n' "$(date -u +%FT%TZ)" "${label}"
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'

  nice -n "${NICE_LEVEL}" "$@" 2>&1 | tee "${log_file}"
  touch "${done_file}"
  printf '[%s] Completed case: %s\n' "$(date -u +%FT%TZ)" "${label}"
}

run_bias_case() {
  local blackjack_pay="$1"
  local payout_label="$2"
  local bias_label="$3"
  local deck="$4"
  local bias="$5"
  local full_policy="${6:-}"
  local case_label="${payout_label}_${bias_label}_d${deck}"
  local output_dir="${SUITE_ROOT}/${payout_label}/${bias_label}_d${deck}"
  local policy_flags=()
  local game_flags=(
    --game blackjack
    --penetration 75
    --ss17 false
    --peek true
    --surr no
    --das true
    --sas 4
    --don ANY
    --rsa false
    --hsa false
    --bj "${blackjack_pay}"
  )

  if [[ -n "${full_policy}" ]]; then
    policy_flags=(--full-policy "${full_policy}")
  fi

  printf 'Bias configuration: payout=%s, %s, deck=%s, bias=%s, zero crossing=%s\n' \
    "${blackjack_pay}" "${bias_label}" "${deck}" "${bias}" \
    "$(bias_threshold "${bias}")"

  run_case "${case_label}" \
    "${COMPARE_BIN}" \
    "${game_flags[@]}" \
    "${TRAINING_FLAGS[@]}" \
    "${EVALUATION_FLAGS[@]}" \
    --decks "${deck}" \
    --count hilo \
    --factor "${HILO_FACTOR}" \
    --bias "${bias}" \
    --continuous-betting-count true \
    --output-dir "${output_dir}" \
    "${policy_flags[@]}"
}

run_payout_suite() {
  local blackjack_pay="$1"
  local payout_label="$2"
  local deck
  local wizard_bias
  local alternating_bias
  local trained_policy

  printf '\n============================================================\n'
  printf 'Starting payout suite: %s (%s)\n' "${payout_label}" "${blackjack_pay}"
  printf '============================================================\n'

  for deck in 1 2 6; do
    wizard_bias="$(wizard_bias_for "${blackjack_pay}" "${deck}")"
    alternating_bias="$(alternating_bias_for "${blackjack_pay}" "${deck}")"

    # Train once with the suite's convergence settings. Training depends on
    # Hi-Lo count states, not on the betting bias, so this table is valid for
    # every bias evaluation below.
    run_bias_case \
      "${blackjack_pay}" "${payout_label}" \
      fixed_0p5pct "${deck}" "${FIXED_0P5_PERCENT_BIAS}"

    trained_policy="${SUITE_ROOT}/${payout_label}/fixed_0p5pct_d${deck}/full_deviations/full_deviations_strategy.json"
    if [[ ! -f "${trained_policy}" ]]; then
      printf 'Expected trained Full-deviation policy was not created: %s\n' \
        "${trained_policy}" >&2
      exit 1
    fi

    run_bias_case \
      "${blackjack_pay}" "${payout_label}" \
      wizard_edge "${deck}" "${wizard_bias}" "${trained_policy}"
    run_bias_case \
      "${blackjack_pay}" "${payout_label}" \
      alternating "${deck}" "${alternating_bias}" "${trained_policy}"
  done
}

main() {
  cd "${PROJECT_DIR}"
  require_executable "${COMPARE_BIN}"
  require_executable "${PROJECT_DIR}/build/bin/EvaluateCountPolicy"
  require_command jq
  require_command awk
  mkdir -p "${LOG_DIR}"

  printf 'Continuous Hi-Lo bias suite: %s\n' "${SUITE_NAME}"
  printf 'Output: %s\n' "${SUITE_ROOT}"
  printf 'Payouts: 3:2 followed by 6:5; threads: %s; seed: %s\n' \
    "${NUM_THREADS}" "${SEED}"
  printf 'Training: diff mode, sample=%s, threshold=%s; configured rounds=%s\n' \
    "${SAMPLE_ROUNDS}" "${DIFF_THRESHOLD}" "${NUM_ROUNDS}"
  printf 'Evaluation: spread=%s; Kelly=%s x %s; range=%s-%s; step=%s\n' \
    "${EVAL_ROUNDS}" "${KELLY_ROUNDS}" "${KELLY_MEASUREMENTS}" \
    "${KELLY_MIN}" "${KELLY_MAX}" "${KELLY_STEP}"
  printf 'Betting count: continuous; Hi-Lo factor: %s\n' "${HILO_FACTOR}"

  run_payout_suite 1.5 bj3-2
  run_payout_suite 1.2 bj6-5

  printf '\n[%s] Bias suite completed: %s\n' \
    "$(date -u +%FT%TZ)" "${SUITE_NAME}"
}

main "$@"
