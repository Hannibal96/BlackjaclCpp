#!/usr/bin/env bash

set -Eeuo pipefail

# Re-evaluate the previously trained Full-deviation Hi-Lo policies while
# limiting cumulative wagers in a round to a fraction of starting bankroll.
#
# The wager cap affects only the Kelly simulation. EvaluateCountPolicy also
# requires a spread simulation, so EVAL_ROUNDS defaults to the smallest valid
# value (one round per worker) rather than repeating the suite's 5-billion-round
# spread measurement for every cap.

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly EVALUATOR="${PROJECT_DIR}/build/bin/EvaluateCountPolicy"
readonly SOURCE_ROOT="${PROJECT_DIR}/paper_materials/performance-suites/h17c03_5b_hilo_bias_continuous"
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-${SOURCE_ROOT}/wizard_edge_wager_caps}"
readonly LOG_DIR="${OUTPUT_ROOT}/logs"

readonly NUM_THREADS="${NUM_THREADS:-10}"
readonly NICE_LEVEL="${NICE_LEVEL:-10}"
readonly SEED="${SEED:-12345}"

readonly EVAL_ROUNDS="${EVAL_ROUNDS:-${NUM_THREADS}}"
readonly KELLY_ROUNDS="${KELLY_ROUNDS:-1000000}"
readonly KELLY_MEASUREMENTS="${KELLY_MEASUREMENTS:-1000}"
readonly KELLY_MIN="${KELLY_MIN:-0.50}"
readonly KELLY_MAX="${KELLY_MAX:-1.00}"
readonly KELLY_STEP="${KELLY_STEP:-0.05}"

readonly HILO_FACTOR=0.005
readonly PENETRATION=75

readonly -a BLACKJACK_PAYOUTS=(1.2 1.5)
readonly -a DECKS=(1 2 6)
readonly -a WAGER_CAPS=(0.1 0.25 0.5 0.75, 1.0)

# Neutral-count player edges in decimal units, matching the Wizard-edge cases
# in the source performance suite.
declare -Ar WIZARD_GAME_BIAS=(
  ["1.5:1"]="-0.0015945"
  ["1.5:2"]="-0.0045688"
  ["1.5:6"]="-0.0063873"
  ["1.2:1"]="-0.0155422"
  ["1.2:2"]="-0.0183038"
  ["1.2:6"]="-0.0199842"
)

require_executable() {
  local path="$1"
  if [[ ! -x "${path}" ]]; then
    printf 'Required executable not found: %s\n' "${path}" >&2
    printf 'Build it first with: cmake --build build -j\n' >&2
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

payout_label_for() {
  case "$1" in
    1.2) printf 'bj6-5\n' ;;
    1.5) printf 'bj3-2\n' ;;
    *)
      printf 'Unsupported blackjack payout: %s\n' "$1" >&2
      return 1
      ;;
  esac
}

cap_label_for() {
  local cap="$1"
  printf '%s\n' "${cap//./p}"
}

strategy_path_for() {
  local payout_label="$1"
  local deck="$2"
  printf '%s/%s/fixed_0p5pct_d%s/full_deviations/full_deviations_strategy.json\n' \
    "${SOURCE_ROOT}" "${payout_label}" "${deck}"
}

wizard_bias_for() {
  local payout="$1"
  local deck="$2"
  local key="${payout}:${deck}"

  if [[ -z "${WIZARD_GAME_BIAS[${key}]+configured}" ]]; then
    printf 'No Wizard-edge bias configured for payout=%s, decks=%s\n' \
      "${payout}" "${deck}" >&2
    return 1
  fi
  printf '%s\n' "${WIZARD_GAME_BIAS[${key}]}"
}

validate_inputs() {
  local payout
  local payout_label
  local deck
  local strategy

  require_executable "${EVALUATOR}"
  require_command jq
  require_command nice
  require_command tee

  if (( EVAL_ROUNDS < NUM_THREADS )); then
    printf 'EVAL_ROUNDS (%s) must be at least NUM_THREADS (%s).\n' \
      "${EVAL_ROUNDS}" "${NUM_THREADS}" >&2
    exit 1
  fi

  for payout in "${BLACKJACK_PAYOUTS[@]}"; do
    payout_label="$(payout_label_for "${payout}")"
    for deck in "${DECKS[@]}"; do
      strategy="$(strategy_path_for "${payout_label}" "${deck}")"
      if [[ ! -f "${strategy}" ]]; then
        printf 'Missing Full-deviation policy: %s\n' "${strategy}" >&2
        exit 1
      fi
    done
  done
}

run_case() {
  local payout="$1"
  local payout_label="$2"
  local deck="$3"
  local cap="$4"
  local cap_label
  local strategy
  local bias
  local case_label
  local output_dir
  local done_file
  local log_file

  cap_label="$(cap_label_for "${cap}")"
  strategy="$(strategy_path_for "${payout_label}" "${deck}")"
  bias="$(wizard_bias_for "${payout}" "${deck}")"
  case_label="${payout_label}_d${deck}_max_wager_${cap_label}"
  output_dir="${OUTPUT_ROOT}/${payout_label}/d${deck}/max_wager_${cap_label}"
  done_file="${LOG_DIR}/${case_label}.done"
  log_file="${LOG_DIR}/${case_label}.console.log"

  if [[ -f "${done_file}" && -f "${output_dir}/results.json" ]]; then
    printf '\n[%s] Skipping completed case: %s\n' \
      "$(date -u +%FT%TZ)" "${case_label}"
    return 0
  fi

  mkdir -p "${output_dir}"
  printf '\n[%s] Starting case: %s\n' "$(date -u +%FT%TZ)" "${case_label}"
  printf '  payout=%s, decks=%s, Wizard bias=%s, max total wager=%s\n' \
    "${payout}" "${deck}" "${bias}" "${cap}"
  printf '  policy=%s\n' "${strategy}"
  printf '  output=%s\n' "${output_dir}"

  nice -n "${NICE_LEVEL}" "${EVALUATOR}" \
    --game blackjack \
    --decks "${deck}" \
    --penetration "${PENETRATION}" \
    --ss17 false \
    --peek true \
    --surr no \
    --das true \
    --sas 4 \
    --don ANY \
    --rsa false \
    --hsa false \
    --bj "${payout}" \
    --count-name hilo \
    --factor "${HILO_FACTOR}" \
    --bias "${bias}" \
    --count-normalization true \
    --continuous-betting-count true \
    --policy-file "${strategy}" \
    --eval-rounds "${EVAL_ROUNDS}" \
    --kelly-rounds "${KELLY_ROUNDS}" \
    --kelly-measurements "${KELLY_MEASUREMENTS}" \
    --kelly-min "${KELLY_MIN}" \
    --kelly-max "${KELLY_MAX}" \
    --kelly-step "${KELLY_STEP}" \
    --max-total-wager-fraction "${cap}" \
    --num-threads "${NUM_THREADS}" \
    --seed "${SEED}" \
    --output-dir "${output_dir}" \
    2>&1 | tee "${log_file}"

  touch "${done_file}"
  printf '[%s] Completed case: %s\n' "$(date -u +%FT%TZ)" "${case_label}"
}

write_summary() {
  local summary_file="${OUTPUT_ROOT}/kelly_summary.csv"
  local temporary_file="${summary_file}.tmp"
  local payout
  local payout_label
  local deck
  local cap
  local cap_label
  local bias
  local result_file

  printf '%s\n' \
    'blackjack_payout,decks,max_total_wager_fraction,wizard_bias,kelly_growth_at_1,kelly_growth_at_1_stddev,empirical_optimum_multiplier,empirical_optimum_growth,mean_total_wager_fraction,max_total_wager_fraction_observed,mean_abs_fx,max_abs_fx,exact_mean_log_increment,quadratic_mean_log_increment,results_file' \
    >"${temporary_file}"

  for payout in "${BLACKJACK_PAYOUTS[@]}"; do
    payout_label="$(payout_label_for "${payout}")"
    for deck in "${DECKS[@]}"; do
      bias="$(wizard_bias_for "${payout}" "${deck}")"
      for cap in "${WAGER_CAPS[@]}"; do
        cap_label="$(cap_label_for "${cap}")"
        result_file="${OUTPUT_ROOT}/${payout_label}/d${deck}/max_wager_${cap_label}/results.json"
        [[ -f "${result_file}" ]] || continue

        jq -r \
          --arg payout "${payout}" \
          --arg deck "${deck}" \
          --arg cap "${cap}" \
          --arg bias "${bias}" \
          --arg result_file "${result_file#${PROJECT_DIR}/}" \
          '[
             $payout,
             $deck,
             $cap,
             $bias,
             .kelly_at_multiplier_1.growth_mean,
             .kelly_at_multiplier_1.growth_stddev,
             .kelly_empirical_optimum.multiplier,
             .kelly_empirical_optimum.growth_mean,
             .kelly_at_multiplier_1.exposure.gross_exposure_summary.mean,
             .kelly_at_multiplier_1.exposure.gross_exposure_summary.maximum,
             .kelly_at_multiplier_1.exposure.absolute_return_summary.mean,
             .kelly_at_multiplier_1.exposure.absolute_return_summary.maximum,
             .kelly_at_multiplier_1.exposure.linearization.exact_mean_log_increment,
             .kelly_at_multiplier_1.exposure.linearization.quadratic_mean_log_increment,
             $result_file
           ] | @csv' \
          "${result_file}" >>"${temporary_file}"
      done
    done
  done

  mv "${temporary_file}" "${summary_file}"
  printf '\nCombined summary: %s\n' "${summary_file}"
}

main() {
  local payout
  local payout_label
  local deck
  local cap

  cd "${PROJECT_DIR}"
  validate_inputs
  mkdir -p "${LOG_DIR}"

  printf 'Wizard-edge maximum-total-wager suite\n'
  printf 'Source policies: %s\n' "${SOURCE_ROOT}"
  printf 'Output: %s\n' "${OUTPUT_ROOT}"
  printf 'Cases: %s payouts x %s decks x %s caps = 24\n' \
    "${#BLACKJACK_PAYOUTS[@]}" "${#DECKS[@]}" "${#WAGER_CAPS[@]}"
  printf 'Kelly: %s rounds x %s measurements, multipliers %s..%s step %s\n' \
    "${KELLY_ROUNDS}" "${KELLY_MEASUREMENTS}" \
    "${KELLY_MIN}" "${KELLY_MAX}" "${KELLY_STEP}"
  printf 'Threads: %s; seed: %s; spread sanity rounds: %s\n' \
    "${NUM_THREADS}" "${SEED}" "${EVAL_ROUNDS}"

  for payout in "${BLACKJACK_PAYOUTS[@]}"; do
    payout_label="$(payout_label_for "${payout}")"
    for deck in "${DECKS[@]}"; do
      for cap in "${WAGER_CAPS[@]}"; do
        run_case "${payout}" "${payout_label}" "${deck}" "${cap}"
      done
    done
  done

  write_summary
  printf '\n[%s] Suite completed.\n' "$(date -u +%FT%TZ)"
}

main "$@"
