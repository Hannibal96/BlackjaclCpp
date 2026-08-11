#!/usr/bin/env bash

set -Eeuo pipefail

# Hi-Lo vs quadratic Kelly vs classical OLS for American H17 blackjack.
# All cases run sequentially so only one simulation uses CPU at a time.
# The order is intentional: finish all Hi-Lo and QK cases before starting OLS.

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly COMPARE_BIN="${PROJECT_DIR}/build/bin/CompareCountStrategies"
readonly ALTERNATING_BIN="${PROJECT_DIR}/build/bin/AlternatingOptimization"

readonly NUM_THREADS=5
readonly NICE_LEVEL="${NICE_LEVEL:-10}"
readonly SUITE_NAME="h17c03_5b"
readonly SUITE_LOG_DIR="${PROJECT_DIR}/checkpoints/performance-suites/${SUITE_NAME}"

readonly NUM_ROUNDS=5000000000
readonly EVAL_ROUNDS=5000000000
readonly GRAPH_ROUNDS=500000000
readonly KELLY_MEASUREMENTS=1000

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
  --bj 1.5
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

  for deck in 1 2 6; do
    run_case "hilo_d${deck}" \
      "${COMPARE_BIN}" \
      "${GAME_FLAGS[@]}" \
      "${RL_FLAGS[@]}" \
      "${EVALUATION_FLAGS[@]}" \
      --decks "${deck}" \
      --count hilo \
      --output-dir \
        "checkpoints/CompareCountStrategies/${SUITE_NAME}_hilo_d${deck}"
  done

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
