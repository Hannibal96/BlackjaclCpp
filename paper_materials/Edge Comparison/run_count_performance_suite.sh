#!/usr/bin/env bash

set -Eeuo pipefail

# Hi-Lo vs quadratic Kelly vs classical OLS for American H17 blackjack.
# All cases run sequentially so only one simulation uses CPU at a time.
# The order is intentional: finish all Hi-Lo and QK cases before starting OLS.

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly COMPARE_BIN="${PROJECT_DIR}/build/bin/CompareCountStrategies"
readonly ALTERNATING_BIN="${PROJECT_DIR}/build/bin/AlternatingOptimization"

readonly NUM_THREADS=5
readonly NICE_LEVEL="${NICE_LEVEL:-10}"
readonly BLACKJACK_PAY="${BLACKJACK_PAY:-1.2}"

case "${BLACKJACK_PAY}" in
  1.2) readonly DEFAULT_SUITE_NAME="h17c03_5b_bj6-5" ;;
  1.5) readonly DEFAULT_SUITE_NAME="h17c03_5b_bj3-2" ;;
  *)
    printf 'Unsupported blackjack payout: %s (expected 1.2 or 1.5)\n' \
      "${BLACKJACK_PAY}" >&2
    exit 1
    ;;
esac

readonly SUITE_NAME="${SUITE_NAME:-${DEFAULT_SUITE_NAME}}"
readonly SUITE_LOG_DIR="${PROJECT_DIR}/checkpoints/performance-suites/${SUITE_NAME}"

readonly NUM_ROUNDS=5000000000
readonly EVAL_ROUNDS=5000000000
readonly GRAPH_ROUNDS=500000000
readonly KELLY_MEASUREMENTS=1000

# Hi-Lo neutral-count player edge in decimal units. These defaults are the
# negatives of the cut-card house edges in json_wod_edge.json for the game
# rules below. Edit this table to configure the bias used for each payout/deck
# combination. Key format: "<blackjack payout>:<decks>".
declare -A HILO_GAME_BIAS=(
  ["1.5:1"]="-0.0015945"
  ["1.5:2"]="-0.0045688"
  ["1.5:6"]="-0.0063873"
  ["1.2:1"]="-0.0155422"
  ["1.2:2"]="-0.0183038"
  ["1.2:6"]="-0.0199842"
)

GAME_FLAGS=(
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
  --bj "${BLACKJACK_PAY}"
)

RL_FLAGS=(
  --num-threads "${NUM_THREADS}"
  --stop-mode diff
  --sample-rounds 100000000
  --diff-threshold 0.005
)

EVALUATION_FLAGS=(
  --num-rounds "${NUM_ROUNDS}"
  --eval-rounds "${EVAL_ROUNDS}"
  --graph-rounds "${GRAPH_ROUNDS}"
  --kelly-measurements "${KELLY_MEASUREMENTS}"
  --kelly-fraction-step 0.05
)

ALTERNATING_FLAGS=(
  --iterations 3
  --sample-every 5
)

require_executable() {
  local path="$1"
  if [[ ! -x "${path}" ]]; then
    printf 'Required executable not found: %s\n' "${path}" >&2
    printf 'Build first with: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j3\n' >&2
    exit 1
  fi
}

hilo_bias_for() {
  local blackjack_pay="$1"
  local deck="$2"
  local payout_key
  local key

  printf -v payout_key '%.1f' "${blackjack_pay}"
  key="${payout_key}:${deck}"

  if [[ -z "${HILO_GAME_BIAS[${key}]+configured}" ]]; then
    printf 'No Hi-Lo game bias configured for payout=%s, decks=%s\n' \
      "${blackjack_pay}" "${deck}" >&2
    return 1
  fi

  printf '%s\n' "${HILO_GAME_BIAS[${key}]}"
}

run_case() {
  local label="$1"
  shift

  local done_file="${SUITE_LOG_DIR}/${label}.done"
  local log_file="${SUITE_LOG_DIR}/${label}.console.log"

  if [[ -f "${done_file}" ]]; then
    printf '\n[%s] Skipping completed case: %s\n' "$(date -u +%FT%TZ)" "${label}"
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

run_alternating_case() {
  local label="$1"
  local objective_flag="$2"
  local constraint_flag="$3"
  local deck="$4"
  local case_label="${label}_d${deck}"
  # AlternatingOptimization appends the actual deck/rule identity to this prefix.
  local checkpoint_name="${SUITE_NAME}_${label}"

  run_case "${case_label}" \
    "${ALTERNATING_BIN}" \
    "${GAME_FLAGS[@]}" \
    "${RL_FLAGS[@]}" \
    "${EVALUATION_FLAGS[@]}" \
    "${ALTERNATING_FLAGS[@]}" \
    --decks "${deck}" \
    "${objective_flag}" \
    "${constraint_flag}" \
    --checkpoint-name "${checkpoint_name}"
}

run_alternating_for_all_decks() {
  local label="$1"
  local objective_flag="$2"
  local constraint_flag="$3"

  local deck
  for deck in 1 2 6; do
    run_alternating_case \
      "${label}" "${objective_flag}" "${constraint_flag}" "${deck}"
  done
}

main() {
  cd "${PROJECT_DIR}"
  require_executable "${COMPARE_BIN}"
  require_executable "${ALTERNATING_BIN}"
  mkdir -p "${SUITE_LOG_DIR}"

  printf 'Performance suite: %s\n' "${SUITE_NAME}"
  printf 'Logs: %s\n' "${SUITE_LOG_DIR}"
  printf 'Threads: %s; nice level: %s\n' "${NUM_THREADS}" "${NICE_LEVEL}"
  printf 'Count-fit rounds: %s; edge/graph rounds: %s\n' \
    "${NUM_ROUNDS}" "${EVAL_ROUNDS}"
  printf 'RL sample rounds: 100000000; diff threshold: 0.03; exploration: defaults\n'
  printf 'Kelly: 1000000 rounds x %s repetitions; multiplier step: 0.05\n' \
    "${KELLY_MEASUREMENTS}"

  local hilo_bias
  # for deck in 1 2 6; do
  #  hilo_bias="$(hilo_bias_for "${BLACKJACK_PAY}" "${deck}")"
  #  printf 'Hi-Lo bias: payout=%s, decks=%s, bias=%s\n' \
  #    "${BLACKJACK_PAY}" "${deck}" "${hilo_bias}"
  #  run_case "hilo_d${deck}" \
  #    "${COMPARE_BIN}" \
  #    "${GAME_FLAGS[@]}" \
  #    "${RL_FLAGS[@]}" \
  #    "${EVALUATION_FLAGS[@]}" \
  #    --decks "${deck}" \
  #    --count hilo \
  #    --bias "${hilo_bias}" \
  #    --output-dir \
  #      "checkpoints/CompareCountStrategies/${SUITE_NAME}_hilo_d${deck}"
  # done

  run_alternating_for_all_decks qk_u \
    --count-quadratic-kelly --count-unconstrained
  run_alternating_for_all_decks qk_s0 \
    --count-quadratic-kelly --count-sum-zero
  run_alternating_for_all_decks qk_b1 \
    --count-quadratic-kelly --count-sum-zero-fixed-b1
  run_alternating_for_all_decks qk_p0 \
    --count-quadratic-kelly --count-sum-zero-fixed-p0-edge

  run_alternating_for_all_decks ols_u \
    --count-classical-ols --count-unconstrained
  run_alternating_for_all_decks ols_s0 \
    --count-classical-ols --count-sum-zero
  run_alternating_for_all_decks ols_b1 \
    --count-classical-ols --count-sum-zero-fixed-b1
  run_alternating_for_all_decks ols_p0 \
    --count-classical-ols --count-sum-zero-fixed-p0-edge

  printf '\n[%s] Performance suite completed: %s\n' \
    "$(date -u +%FT%TZ)" "${SUITE_NAME}"
}

main "$@"
