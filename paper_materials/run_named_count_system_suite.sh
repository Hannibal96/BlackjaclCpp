#!/usr/bin/env bash

set -Eeuo pipefail

# Compare every named blackjack count under Basic strategy and freshly trained
# Full deviations. The Python comparison app delegates both policies to the
# authoritative EvaluateCountPolicy executable.
#
# Default matrix:
#   2 blackjack payouts x 3 deck counts x 7 named counts x 2 policies
#   = 84 policy evaluations (42 compare-app invocations).
#
# Important count semantics:
#   * Every named count uses its rank tags from CountingMethods.h.
#   * The betting intercept is overridden with the rule/deck-specific Wizard
#     edge. Level-one counts use a 0.005 signal slope; the level-two Hi-Opt II,
#     Omega II, and Zen counts use 0.0025 so their doubled tags do not double
#     the betting signal. Kelly multipliers scale that complete signal.
#   * Betting uses the continuous selected count by default, while Full-
#     deviation playing states retain the app's integer count resolution.
#   * Named KO is an unnormalized running count with an IRC of zero. This keeps
#     c=0 at the shuffle, so the Wizard bias is the initial betting signal.
#   * KO uses a wider Kelly-multiplier search because its unnormalized running
#     count has a materially different scale from the true-counted systems.

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly COMPARE_APP="${PROJECT_DIR}/scripts/compare_count_strategies.py"
readonly EVALUATOR="${PROJECT_DIR}/build/bin/EvaluateCountPolicy"

readonly SUITE_NAME="${SUITE_NAME:-h17c03_5b_named_counts_rescaled}"
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/paper_materials/performance-suites/${SUITE_NAME}}"
readonly LOG_DIR="${OUTPUT_ROOT}/logs"

readonly NUM_THREADS="${NUM_THREADS:-10}"
readonly NICE_LEVEL="${NICE_LEVEL:-10}"
readonly SEED="${SEED:-12345}"

readonly TRAINING_ROUNDS="${TRAINING_ROUNDS:-5000000000}"
readonly TRAINING_STOP_MODE="${TRAINING_STOP_MODE:-diff}"
readonly SAMPLE_ROUNDS="${SAMPLE_ROUNDS:-100000000}"
readonly DIFF_THRESHOLD="${DIFF_THRESHOLD:-0.005}"

readonly EVAL_ROUNDS="${EVAL_ROUNDS:-5000000000}"
readonly KELLY_ROUNDS="${KELLY_ROUNDS:-1000000}"
readonly KELLY_MEASUREMENTS="${KELLY_MEASUREMENTS:-1000}"
readonly KELLY_MIN="${KELLY_MIN:-0.50}"
readonly KELLY_MAX="${KELLY_MAX:-1.00}"
readonly KELLY_STEP="${KELLY_STEP:-0.05}"
readonly KO_KELLY_MIN="${KO_KELLY_MIN:-0.25}"
readonly KO_KELLY_MAX="${KO_KELLY_MAX:-2.50}"
readonly MAX_TOTAL_WAGER_FRACTION="${MAX_TOTAL_WAGER_FRACTION:-1.0}"

readonly COUNT_FACTOR="${COUNT_FACTOR:-0.005}"
readonly DOUBLED_COUNT_FACTOR="${DOUBLED_COUNT_FACTOR:-0.0025}"
readonly CONTINUOUS_BETTING_COUNT="${CONTINUOUS_BETTING_COUNT:-true}"
readonly COUNT_MIN="${COUNT_MIN:--5}"
readonly COUNT_MAX="${COUNT_MAX:-5}"
readonly KO_COUNT_MIN="${KO_COUNT_MIN:--20}"
readonly KO_COUNT_MAX="${KO_COUNT_MAX:-30}"
readonly PENETRATION="${PENETRATION:-75}"

readonly -a BLACKJACK_PAYOUTS=(1.5 1.2)
readonly -a DECKS=(1 2 6)

# Override, for example, with:
#   COUNTS='hilo halves ko' ./paper_materials/run_named_count_system_suite.sh
readonly COUNTS_TEXT="${COUNTS:-hilo halves ko hiopt1 hiopt2 omega2 zen}"
read -r -a COUNT_SYSTEMS <<<"${COUNTS_TEXT}"
readonly -a COUNT_SYSTEMS

# Neutral-count player edges for the suite rules, in decimal units. These are
# the same Wizard-edge values used in the existing Hi-Lo performance suite.
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
    1.5) printf 'bj3-2\n' ;;
    1.2) printf 'bj6-5\n' ;;
    *)
      printf 'Unsupported blackjack payout: %s\n' "$1" >&2
      return 1
      ;;
  esac
}

wizard_bias_for() {
  local payout="$1"
  local deck="$2"
  local key="${payout}:${deck}"

  if [[ -z "${WIZARD_GAME_BIAS[${key}]+configured}" ]]; then
    printf 'No Wizard edge configured for payout=%s, decks=%s\n' \
      "${payout}" "${deck}" >&2
    return 1
  fi
  printf '%s\n' "${WIZARD_GAME_BIAS[${key}]}"
}

validate_count_name() {
  case "$1" in
    none|hilo|halves|ko|hiopt1|hiopt2|omega2|zen) return 0 ;;
    *)
      printf 'Unsupported named count: %s\n' "$1" >&2
      printf 'Supported names: none hilo halves ko hiopt1 hiopt2 omega2 zen\n' >&2
      return 1
      ;;
  esac
}

validate_configuration() {
  local count

  require_executable "${COMPARE_APP}"
  require_executable "${EVALUATOR}"
  require_command jq
  require_command nice
  require_command tee

  if (( ${#COUNT_SYSTEMS[@]} == 0 )); then
    printf 'COUNTS must contain at least one named count.\n' >&2
    exit 1
  fi
  for count in "${COUNT_SYSTEMS[@]}"; do
    validate_count_name "${count}"
  done

  if (( EVAL_ROUNDS < NUM_THREADS || KELLY_ROUNDS < NUM_THREADS )); then
    printf 'Evaluation and Kelly rounds must each be at least NUM_THREADS (%s).\n' \
      "${NUM_THREADS}" >&2
    exit 1
  fi
  if [[ "${TRAINING_STOP_MODE}" != "diff" && \
        "${TRAINING_STOP_MODE}" != "rounds" ]]; then
    printf 'TRAINING_STOP_MODE must be diff or rounds.\n' >&2
    exit 1
  fi
}

run_case() {
  local label="$1"
  local expected_result="$2"
  shift 2

  local done_file="${LOG_DIR}/${label}.done"
  local log_file="${LOG_DIR}/${label}.console.log"

  if [[ -f "${done_file}" && -f "${expected_result}" ]]; then
    printf '\n[%s] Skipping completed case: %s\n' \
      "$(date -u +%FT%TZ)" "${label}"
    return 0
  fi

  printf '\n[%s] Starting case: %s\n' "$(date -u +%FT%TZ)" "${label}"
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'

  nice -n "${NICE_LEVEL}" "$@" 2>&1 | tee "${log_file}"

  if [[ ! -f "${expected_result}" ]]; then
    printf 'Expected result was not created: %s\n' "${expected_result}" >&2
    return 1
  fi
  touch "${done_file}"
  printf '[%s] Completed case: %s\n' "$(date -u +%FT%TZ)" "${label}"
}

run_count_case() {
  local payout="$1"
  local payout_label="$2"
  local deck="$3"
  local count="$4"
  local bias
  local case_label
  local output_dir
  local selected_count_min="${COUNT_MIN}"
  local selected_count_max="${COUNT_MAX}"
  local selected_factor="${COUNT_FACTOR}"
  local selected_kelly_min="${KELLY_MIN}"
  local selected_kelly_max="${KELLY_MAX}"
  local -a normalization_flags=(--count-normalization true)

  bias="$(wizard_bias_for "${payout}" "${deck}")"
  case_label="${payout_label}_${count}_d${deck}"
  output_dir="${OUTPUT_ROOT}/${payout_label}/${count}_d${deck}"

  if [[ "${count}" == "ko" ]]; then
    normalization_flags=(
      --count-normalization running
      --initial-count 0
      --initial-count-per-deck 0
    )
    selected_count_min="${KO_COUNT_MIN}"
    selected_count_max="${KO_COUNT_MAX}"
    selected_kelly_min="${KO_KELLY_MIN}"
    selected_kelly_max="${KO_KELLY_MAX}"
  elif [[ "${count}" == "hiopt2" || "${count}" == "omega2" || \
          "${count}" == "zen" ]]; then
    selected_factor="${DOUBLED_COUNT_FACTOR}"
  fi

  printf '\nConfiguration: payout=%s, decks=%s, count=%s, factor=%s, bias=%s, count-range=%s..%s, Kelly-range=%s..%s\n' \
    "${payout}" "${deck}" "${count}" "${selected_factor}" "${bias}" \
    "${selected_count_min}" "${selected_count_max}" \
    "${selected_kelly_min}" "${selected_kelly_max}"

  run_case "${case_label}" "${output_dir}/results.json" \
    "${COMPARE_APP}" \
    --game blackjack \
    --count-name "${count}" \
    --skip-illustrious18 \
    --output-dir "${output_dir}" \
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
    --factor "${selected_factor}" \
    --bias "${bias}" \
    "${normalization_flags[@]}" \
    --continuous-betting-count "${CONTINUOUS_BETTING_COUNT}" \
    --min-count "${selected_count_min}" \
    --max-count "${selected_count_max}" \
    --num-rounds "${TRAINING_ROUNDS}" \
    --stop-mode "${TRAINING_STOP_MODE}" \
    --sample-rounds "${SAMPLE_ROUNDS}" \
    --diff-threshold "${DIFF_THRESHOLD}" \
    --eval-rounds "${EVAL_ROUNDS}" \
    --kelly-rounds "${KELLY_ROUNDS}" \
    --kelly-measurements "${KELLY_MEASUREMENTS}" \
    --kelly-min "${selected_kelly_min}" \
    --kelly-max "${selected_kelly_max}" \
    --kelly-step "${KELLY_STEP}" \
    --max-total-wager-fraction "${MAX_TOTAL_WAGER_FRACTION}" \
    --num-threads "${NUM_THREADS}" \
    --seed "${SEED}"
}

write_summary() {
  local summary_file="${OUTPUT_ROOT}/named_count_policy_summary.csv"
  local temporary_file="${summary_file}.tmp"
  local payout
  local payout_label
  local deck
  local count
  local policy
  local result_file

  printf '%s\n' \
    'blackjack_payout,decks,count,policy,count_normalization,initial_count,initial_count_per_deck,count_minimum,count_maximum,factor,bias,continuous_betting_count,spread_edge,spread_stddev,kelly_growth_at_1,kelly_growth_stddev_at_1,empirical_optimum_multiplier,empirical_optimum_growth,mean_total_wager_fraction,max_total_wager_fraction,mean_abs_fx,max_abs_fx,result_file' \
    >"${temporary_file}"

  for payout in "${BLACKJACK_PAYOUTS[@]}"; do
    payout_label="$(payout_label_for "${payout}")"
    for deck in "${DECKS[@]}"; do
      for count in "${COUNT_SYSTEMS[@]}"; do
        for policy in basic full_deviations; do
          result_file="${OUTPUT_ROOT}/${payout_label}/${count}_d${deck}/${policy}/results.json"
          [[ -f "${result_file}" ]] || continue

          jq -r \
            --arg payout "${payout}" \
            --arg deck "${deck}" \
            --arg count_name "${count}" \
            --arg policy_name "${policy}" \
            --arg result_file "${result_file#${PROJECT_DIR}/}" \
            '[
               $payout,
               $deck,
               $count_name,
               $policy_name,
               .count.count_normalization,
               .count.initial_count,
               .count.initial_count_per_deck,
               .count.min_count,
               .count.max_count,
               .count.factor,
               .count.bias,
               .count.continuous_betting_count,
               .spread.edge_per_round,
               .spread.outcome_stddev,
               .kelly_at_multiplier_1.growth_mean,
               .kelly_at_multiplier_1.growth_stddev,
               .kelly_empirical_optimum.multiplier,
               .kelly_empirical_optimum.growth_mean,
               .kelly_at_multiplier_1.exposure.gross_exposure_summary.mean,
               .kelly_at_multiplier_1.exposure.gross_exposure_summary.maximum,
               .kelly_at_multiplier_1.exposure.absolute_return_summary.mean,
               .kelly_at_multiplier_1.exposure.absolute_return_summary.maximum,
               $result_file
             ] | @csv' \
            "${result_file}" >>"${temporary_file}"
        done
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
  local count
  local compare_cases
  local policy_evaluations

  cd "${PROJECT_DIR}"
  validate_configuration
  mkdir -p "${LOG_DIR}"

  compare_cases=$((${#BLACKJACK_PAYOUTS[@]} * ${#DECKS[@]} * ${#COUNT_SYSTEMS[@]}))
  policy_evaluations=$((compare_cases * 2))

  printf 'Named-count policy suite: %s\n' "${SUITE_NAME}"
  printf 'Output: %s\n' "${OUTPUT_ROOT}"
  printf 'Counts: %s\n' "${COUNT_SYSTEMS[*]}"
  printf 'Matrix: %s compare-app cases; %s policy evaluations\n' \
    "${compare_cases}" "${policy_evaluations}"
  printf 'Policies: Basic and freshly trained Full deviations\n'
  printf 'Rules: H17, peek, no surrender, DAS, 4 hands, 75%% penetration\n'
  printf 'Training: mode=%s, configured rounds=%s, sample=%s, diff=%s\n' \
    "${TRAINING_STOP_MODE}" "${TRAINING_ROUNDS}" \
    "${SAMPLE_ROUNDS}" "${DIFF_THRESHOLD}"
  printf 'Evaluation: spread=%s; Kelly=%s x %s; range=%s..%s step=%s\n' \
    "${EVAL_ROUNDS}" "${KELLY_ROUNDS}" "${KELLY_MEASUREMENTS}" \
    "${KELLY_MIN}" "${KELLY_MAX}" "${KELLY_STEP}"
  printf 'Count factors: level-one=%s; Hi-Opt II/Omega II/Zen=%s\n' \
    "${COUNT_FACTOR}" "${DOUBLED_COUNT_FACTOR}"
  printf 'Kelly multipliers: default=%s..%s; KO=%s..%s; step=%s\n' \
    "${KELLY_MIN}" "${KELLY_MAX}" "${KO_KELLY_MIN}" \
    "${KO_KELLY_MAX}" "${KELLY_STEP}"
  printf 'KO: zero-IRC running count with playing-state range %s..%s\n' \
    "${KO_COUNT_MIN}" "${KO_COUNT_MAX}"

  for payout in "${BLACKJACK_PAYOUTS[@]}"; do
    payout_label="$(payout_label_for "${payout}")"
    for deck in "${DECKS[@]}"; do
      for count in "${COUNT_SYSTEMS[@]}"; do
        run_count_case "${payout}" "${payout_label}" "${deck}" "${count}"
      done
    done
  done

  write_summary
  printf '\n[%s] Named-count suite completed.\n' "$(date -u +%FT%TZ)"
}

main "$@"
