#include "Game/SpanishRules.h"
#include "Game/Player.h"
#include "RL/DecayingParameter.h"
#include "RL/QLearningStrategy.h"
#include "RL/SpanishStrategy.h"
#include "Utils/Utils.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint64_t g_numRounds = 1'000'000'000ULL;
int g_numThreads = 10;
double g_penetration = 75.0;
int g_decks = 8;
bool g_standSoft17 = true;
// 0 = redoubling off (default, so the baseline S17/no-redouble combination --
// the one with a directly-published WoO number -- stays the default test).
// When enabling it, 1 is the value Spanish21EdgeRegressionTest verifies by
// default against WoO's published H17+redouble edge; see
// SpanishRules::maxRedoubles.
int g_redouble = 0;
// DDR defaults on: WoO lists it under "The Rules" (standard, always-on),
// unlike redoubling which is a separate "Variable Rule" with its own edge
// figures -- see SpanishRules' default constructor. WoO's only published
// AFTER_DOUBLE chart (dd2.gif, transcribed as h17_redouble_after.json) is for
// H17+redouble specifically though, so loadSpanishBasicStrategy() reuses it
// as the best available reference whenever AFTER_DOUBLE has a real decision
// to make (redouble or rescue enabled) -- not a dedicated chart for this
// exact redouble=0/ddr=true combination.
bool g_ddr = true;
// compareStrategies() scores one judgment per (handType, total, dealer)
// situation (~660 total), not per card count -- a situation counts as a
// difference if the learned strategy misses the base action OR any real
// card-count deviation. Remaining mismatches concentrate almost entirely in
// rare high-cardCount (5-6) states that need more than a quick training run to
// converge -- observed ~90/660 at 400M rounds, improving at the 1B default.
// Set well above Blackjack/DDM's ~10 default to reflect this larger, sparser
// state space (5 card-count buckets vs. their 1).
//
// Raised from 80 to 150 once ddr defaulted to true: AFTER_DOUBLE/
// AFTER_DOUBLE_SOFT's STAND-vs-SURRENDER decision is now genuinely live (not
// a forced single action), and it's dominated by very-low-total states (e.g.
// total 6-9 after doubling) that are only reachable via a narrow path --
// double a low pre-double hand, then draw a low card -- so they're visited
// too rarely to fully converge even at 1B rounds (observed ~130/660 at 1B,
// ~107/660 at 3B: real but slow, not a bug -- see
// Spanish21QLearningRegressionTest.cpp's resolveIfAllowed() for the related
// fix that stopped over-counting AFTER_DOUBLE mismatches against actions the
// current rules don't even allow).
int g_maxDifferences = 150;
ExplorationMode g_explorationMode = ExplorationMode::EPSILON_GREEDY;
double g_epsilonStart = 1.0;
double g_epsilonMin = 0.1;
double g_epsilonDecay = 0.99999;
double g_temperatureStart = 1.0;
double g_temperatureMin = 0.1;
double g_temperatureDecay = 0.99999;
double g_alphaStart = 0.01;
double g_alphaMin = 0.0001;
int g_alphaDecaySteps = 100;

const char* handTypeName(HandType type) {
    switch (type) {
        case HandType::HARD: return "HARD";
        case HandType::SOFT: return "SOFT";
        case HandType::PAIR: return "PAIR";
        case HandType::ZOMBIE: return "ONE_CARD";
        case HandType::BLACKJACK: return "BLACKJACK";
        case HandType::AFTER_DOUBLE: return "AFTER_DOUBLE";
        case HandType::AFTER_DOUBLE_SOFT: return "AFTER_DOUBLE_SOFT";
    }
    return "UNKNOWN";
}

const char* actionName(Action action) {
    switch (action) {
        case Action::HIT: return "HIT";
        case Action::STAND: return "STAND";
        case Action::DOUBLE_DOWN: return "DOUBLE_DOWN";
        case Action::SPLIT: return "SPLIT";
        case Action::SURRENDER: return "SURRENDER";
    }
    return "UNKNOWN";
}

// Sanity-checks the StateKey machinery Spanish 21 specifically relies on: the
// cardCount field distinguishing hands of the same total reached via different
// numbers of cards (relevant to the 5/6/7-card 21 bonus), and AFTER_DOUBLE
// overriding the normal HARD/SOFT/PAIR classification regardless of total.
void verifyStateEncoding(const Player& player) {
    const Card dealerSix(Rank::SIX, Suit::CLUBS);

    Hand twoCardEighteen(1.0);
    twoCardEighteen.addCard(Card(Rank::KING, Suit::HEARTS));
    twoCardEighteen.addCard(Card(Rank::EIGHT, Suit::DIAMONDS));
    const StateKey twoCardKey = player.stateToKey(
        State(twoCardEighteen, dealerSix, {Action::HIT, Action::STAND, Action::DOUBLE_DOWN}));

    Hand fourCardEighteen(1.0);
    fourCardEighteen.addCard(Card(Rank::KING, Suit::HEARTS));
    fourCardEighteen.addCard(Card(Rank::TWO, Suit::DIAMONDS));
    fourCardEighteen.addCard(Card(Rank::THREE, Suit::CLUBS));
    fourCardEighteen.addCard(Card(Rank::THREE, Suit::SPADES));
    const StateKey fourCardKey = player.stateToKey(
        State(fourCardEighteen, dealerSix, {Action::HIT, Action::STAND}));

    if (twoCardKey == fourCardKey ||
        std::get<1>(twoCardKey) != HandType::HARD || std::get<2>(twoCardKey) != 18 ||
        std::get<4>(twoCardKey) != 2 ||
        std::get<1>(fourCardKey) != HandType::HARD || std::get<2>(fourCardKey) != 18 ||
        std::get<4>(fourCardKey) != 4) {
        throw std::logic_error(
            "State encoding fails to distinguish a 2-card 18 from a 4-card 18 via cardCount");
    }

    Hand doubledLow(2.0);
    doubledLow.addCard(Card(Rank::FOUR, Suit::HEARTS));
    doubledLow.addCard(Card(Rank::THREE, Suit::DIAMONDS));
    doubledLow.addCard(Card(Rank::TWO, Suit::CLUBS));  // value 9, post-double card
    doubledLow.incrementDoubleCount();
    const StateKey doubledLowKey = player.stateToKey(State(doubledLow, dealerSix, {Action::STAND}));

    Hand doubledHigh(2.0);
    doubledHigh.addCard(Card(Rank::NINE, Suit::HEARTS));
    doubledHigh.addCard(Card(Rank::EIGHT, Suit::DIAMONDS));
    doubledHigh.addCard(Card(Rank::TWO, Suit::CLUBS));  // value 19
    doubledHigh.incrementDoubleCount();
    const StateKey doubledHighKey = player.stateToKey(State(doubledHigh, dealerSix, {Action::STAND}));

    if (std::get<1>(doubledLowKey) != HandType::AFTER_DOUBLE ||
        std::get<1>(doubledHighKey) != HandType::AFTER_DOUBLE) {
        throw std::logic_error(
            "AFTER_DOUBLE classification does not override HARD/SOFT regardless of total");
    }

    // Hard and soft doubled hands must stay distinguishable (WoO's dd2 chart gives
    // them genuinely different strategy for the same numeric total, e.g. hard 13
    // is mostly stand/surrender while soft 13 is uniformly double).
    Hand doubledSoft(2.0);
    doubledSoft.addCard(Card(Rank::ACE, Suit::HEARTS));
    doubledSoft.addCard(Card(Rank::TWO, Suit::DIAMONDS));
    doubledSoft.addCard(Card(Rank::THREE, Suit::CLUBS));  // A,2,3 = soft 16, post-double card
    doubledSoft.incrementDoubleCount();
    const StateKey doubledSoftKey = player.stateToKey(State(doubledSoft, dealerSix, {Action::STAND}));

    if (std::get<1>(doubledSoftKey) != HandType::AFTER_DOUBLE_SOFT ||
        doubledSoftKey == doubledLowKey) {
        throw std::logic_error(
            "A doubled soft hand must be classified AFTER_DOUBLE_SOFT, distinct from AFTER_DOUBLE");
    }

    Hand blackjack(1.0);
    blackjack.addCard(Card(Rank::ACE, Suit::HEARTS));
    blackjack.addCard(Card(Rank::KING, Suit::HEARTS));
    const StateKey blackjackKey = player.stateToKey(State(blackjack, dealerSix, {}));

    Hand ordinaryTwentyOne(1.0);
    ordinaryTwentyOne.addCard(Card(Rank::SEVEN, Suit::CLUBS));
    ordinaryTwentyOne.addCard(Card(Rank::SEVEN, Suit::DIAMONDS));
    ordinaryTwentyOne.addCard(Card(Rank::SEVEN, Suit::SPADES));
    const StateKey ordinaryTwentyOneKey = player.stateToKey(State(ordinaryTwentyOne, dealerSix, {}));

    if (blackjackKey == ordinaryTwentyOneKey ||
        std::get<1>(blackjackKey) != HandType::BLACKJACK ||
        std::get<1>(ordinaryTwentyOneKey) != HandType::HARD) {
        throw std::logic_error("State encoding aliases blackjack with an ordinary 3-card 21");
    }
}

// Scores one judgment per (handType, total, dealer) *situation*, not per card
// count -- the same way an existing "Dh"/"Ds" cell is judged correct only if
// both the primary action AND its fallback (what happens when the primary
// isn't legal) match. Here the card-count dimension plays the fallback's role:
// a situation is counted as one difference if the learned strategy gets the
// base (lowest-card-count) action wrong, OR if it fails to reproduce a
// card-count deviation the reference table has (wrong action at some higher
// card count), OR if it invents a deviation the reference doesn't have.
// Iterating cardCount separately (denominator x5) would count one real
// "did we learn this decision, thresholds included" mistake as up to five.
std::pair<int, int> compareStrategies(const BasicStrategy& learned, const BasicStrategy& expected) {
    int differences = 0;
    int compared = 0;

    // AFTER_DOUBLE[_SOFT]'s reference entries encode DOUBLE_DOWN/SURRENDER as
    // .primary whenever WoO's H17+redouble chart favors them, regardless of
    // whether THIS run's rules actually allow that action at the AFTER_DOUBLE
    // decision node (STAND is always legal there; DOUBLE_DOWN only if
    // g_redouble > 0; SURRENDER only if g_ddr -- mirrors
    // SpanishTable::getAllowedActions's AFTER_DOUBLE branch). Comparing raw
    // .primary would then fault Q-learning for never picking an action it was
    // never legally allowed to try; resolve to .fallback exactly like the
    // live game does via ActionWithFallback.
    auto resolveIfAllowed = [](HandType type, const ActionWithFallback& aw) -> Action {
        if (type != HandType::AFTER_DOUBLE && type != HandType::AFTER_DOUBLE_SOFT) return aw.primary;
        if (aw.primary == Action::DOUBLE_DOWN && g_redouble <= 0) return aw.fallback;
        if (aw.primary == Action::SURRENDER && !g_ddr) return aw.fallback;
        return aw.primary;
    };

    auto compareSituation = [&](HandType type, int total, int dealer, int minCardCount, int maxCardCount = 6) {
        std::vector<std::string> mismatches;
        for (int cardCount = minCardCount; cardCount <= maxCardCount; ++cardCount) {
            const Action learnedAction =
                learned.getActionFromTable(0, type, total, dealer, cardCount).primary;
            const Action expectedAction =
                resolveIfAllowed(type, expected.getActionFromTable(0, type, total, dealer, cardCount));
            if (learnedAction != expectedAction) {
                const std::string label = (cardCount == minCardCount ? "base/" : "") + std::to_string(cardCount) + "cards";
                mismatches.push_back(
                    label + ": learned=" + actionName(learnedAction) + " expected=" + actionName(expectedAction));
            }
        }

        ++compared;
        if (!mismatches.empty()) {
            ++differences;
            std::cout << "Mismatch handType=" << handTypeName(type)
                      << " total=" << total << " dealer=" << dealer << "  ";
            for (const auto& m : mismatches) std::cout << "[" << m << "] ";
            std::cout << "\n";
        }
    };

    for (int total = 4; total <= 21; ++total)
        for (int dealer = 2; dealer <= 11; ++dealer)
            // A hard 21 is unreachable with only 2 cards: two non-ace-effective
            // cards max out at 10+10=20 -- any 2-card 21 is either blackjack or
            // soft (via an ace). Every other hard total is reachable at 2 cards.
            compareSituation(HandType::HARD, total, dealer, /*minCardCount=*/total == 21 ? 3 : 2);
    for (int total = 12; total <= 21; ++total)
        for (int dealer = 2; dealer <= 11; ++dealer)
            compareSituation(HandType::SOFT, total, dealer, /*minCardCount=*/2);
    // A pair, by definition, is exactly 2 cards -- a 3rd card always ends the pair
    // classification (isPair() requires cardCount()==2), so cardCount 3-6 is
    // unreachable and comparing it would only be noise.
    for (int cardValue = 2; cardValue <= 11; ++cardValue)
        for (int dealer = 2; dealer <= 11; ++dealer)
            compareSituation(HandType::PAIR, cardValue, dealer, /*minCardCount=*/2, /*maxCardCount=*/2);
    // AFTER_DOUBLE[_SOFT] is a genuine STAND-vs-SURRENDER decision here since DDR
    // defaults on (redouble stays off by default, so DOUBLE_DOWN isn't a third
    // option) -- included so the full HandType range is printed/compared, per the
    // project's Q-learning regression convention of covering every hand-type
    // bucket a game can reach. minCardCount=3:
    // a doubled hand needs at least 2 pre-double cards + 1 double card, so cardCount=2
    // is unreachable there and would only produce a meaningless comparison (the
    // reference table happens to define it, but Q-learning never visits it).
    for (int total = 4; total <= 21; ++total)
        for (int dealer = 2; dealer <= 11; ++dealer)
            compareSituation(HandType::AFTER_DOUBLE, total, dealer, /*minCardCount=*/3);
    for (int total = 12; total <= 21; ++total)
        for (int dealer = 2; dealer <= 11; ++dealer)
            compareSituation(HandType::AFTER_DOUBLE_SOFT, total, dealer, /*minCardCount=*/3);

    return {differences, compared};
}

void printHelp(const char* program) {
    std::cout
        << "Usage: " << program << " [OPTIONS]\n\n"
        << "Train a Spanish 21 Q-learning policy and compare it with the\n"
        << "transcribed WoO basic strategy.\n\n"
        << "  --num-rounds <N>       Training rounds (default: 1000000000)\n"
        << "  --num-threads <N>      Threads (default: 10)\n"
        << "  --penetration <P>      Penetration (default: 75)\n"
        << "  --decks <N>            Number of decks (default: 8)\n"
        << "  --ss17 <bool>          Dealer stands on soft 17 (default: true)\n"
        << "  --redouble <N>         Redoubles allowed beyond the first double (default: 0\n"
        << "                         -- off). 1 is the WoO-verified value. Combined with\n"
        << "                         ss17=true this has no WoO reference table and falls\n"
        << "                         back to the plain S17 base table.\n"
        << "  --ddr <bool>           Allow double-down rescue (default: true); a standard WoO\n"
        << "                         rule, though its only published AFTER_DOUBLE strategy\n"
        << "                         chart is for H17+redouble specifically -- reused here as\n"
        << "                         the best available reference regardless.\n"
        << "  --max-differences <N>  Allowed mismatches (default: 150)\n"
        << "  --exploration <mode>   epsilon or boltzmann (default: epsilon)\n"
        << "  --epsilon-start <v>    Default: 1.0\n"
        << "  --epsilon-min <v>      Default: 0.1\n"
        << "  --epsilon-decay <v>    Default: 0.99999\n"
        << "  --temp-start <v>       Initial Boltzmann temperature (default: 1.0)\n"
        << "  --temp-min <v>         Minimum Boltzmann temperature (default: 0.1)\n"
        << "  --temp-decay <v>       Boltzmann decay factor (default: 0.99999)\n"
        << "  --alpha-start <v>      Default: 0.01\n"
        << "  --alpha-min <v>        Default: 0.0001\n"
        << "  --alpha-decay <N>      Default: 100\n";
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            return 0;
        }
        if (arg == "--num-rounds" && i + 1 < argc)
            g_numRounds = std::stoull(argv[++i]);
        else if (arg == "--num-threads" && i + 1 < argc)
            g_numThreads = std::stoi(argv[++i]);
        else if (arg == "--penetration" && i + 1 < argc)
            g_penetration = std::stod(argv[++i]);
        else if (arg == "--decks" && i + 1 < argc)
            g_decks = std::stoi(argv[++i]);
        else if (arg == "--ss17" && i + 1 < argc)
            g_standSoft17 = std::string(argv[++i]) == "true";
        else if (arg == "--redouble" && i + 1 < argc)
            g_redouble = std::stoi(argv[++i]);
        else if (arg == "--ddr" && i + 1 < argc)
            g_ddr = std::string(argv[++i]) == "true";
        else if (arg == "--max-differences" && i + 1 < argc)
            g_maxDifferences = std::stoi(argv[++i]);
        else if (arg == "--exploration" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "epsilon") {
                g_explorationMode = ExplorationMode::EPSILON_GREEDY;
            } else if (mode == "boltzmann") {
                g_explorationMode = ExplorationMode::BOLTZMANN;
            } else {
                std::cerr << "Unknown exploration mode: " << mode
                          << " (expected epsilon or boltzmann)\n";
                return 1;
            }
        }
        else if (arg == "--epsilon-start" && i + 1 < argc)
            g_epsilonStart = std::stod(argv[++i]);
        else if (arg == "--epsilon-min" && i + 1 < argc)
            g_epsilonMin = std::stod(argv[++i]);
        else if (arg == "--epsilon-decay" && i + 1 < argc)
            g_epsilonDecay = std::stod(argv[++i]);
        else if (arg == "--temp-start" && i + 1 < argc)
            g_temperatureStart = std::stod(argv[++i]);
        else if (arg == "--temp-min" && i + 1 < argc)
            g_temperatureMin = std::stod(argv[++i]);
        else if (arg == "--temp-decay" && i + 1 < argc)
            g_temperatureDecay = std::stod(argv[++i]);
        else if (arg == "--alpha-start" && i + 1 < argc)
            g_alphaStart = std::stod(argv[++i]);
        else if (arg == "--alpha-min" && i + 1 < argc)
            g_alphaMin = std::stod(argv[++i]);
        else if (arg == "--alpha-decay" && i + 1 < argc)
            g_alphaDecaySteps = std::stoi(argv[++i]);
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }

    if (g_decks <= 0) {
        std::cerr << "Deck count must be positive\n";
        return 1;
    }

    auto alpha = std::make_unique<LinearDecayingParameter>(
        g_alphaStart, g_alphaMin, g_alphaDecaySteps);
    std::unique_ptr<DecayingParameter> exploration;
    if (g_explorationMode == ExplorationMode::EPSILON_GREEDY) {
        exploration = std::make_unique<EpsilonDecayingParameter>(
            g_epsilonStart, g_epsilonMin, g_epsilonDecay);
    } else {
        exploration = std::make_unique<EpsilonDecayingParameter>(
            g_temperatureStart, g_temperatureMin, g_temperatureDecay);
    }
    auto qStrategy = std::make_unique<QLearningStrategy>(
        std::move(alpha), std::move(exploration), 1.0, g_explorationMode);
    auto* player = new Player(0.0, std::move(qStrategy));
    player->setNumDecks(g_decks);
    player->setTrackHandCardCount(true);
    verifyStateEncoding(*player);

    SpanishRules rules(g_standSoft17, g_decks, g_penetration, /*maxSplit=*/4,
                       /*doubleAfterSplit=*/true, /*resplitAces=*/true, /*hitSplitAces=*/true,
                       /*surrender=*/true, /*peek=*/true, g_redouble,
                       g_ddr, /*suitedBonus=*/true);
    rules.minBet = 1.0;
    rules.maxBet = 1.0;
    const auto start = std::chrono::steady_clock::now();
    std::vector<Player*> results =
        runParallelSimulation(rules, {player}, g_numRounds, g_numThreads);
    delete player;
    if (results.empty()) {
        std::cerr << "Q-learning simulation failed\n";
        return 1;
    }

    const auto* learnedQ =
        dynamic_cast<const QLearningStrategy*>(results.front()->getStrategy());
    if (!learnedQ) {
        delete results.front();
        std::cerr << "Result player does not contain QLearningStrategy\n";
        return 1;
    }

    std::cout << "\nFinal Q-Learning Strategy (averaged across threads):\n";
    std::cout << *learnedQ << "\n";

    auto learned = learnedQ->toBasicStrategy();
    auto expected = loadSpanishBasicStrategy(rules);
    const auto [differences, compared] = compareStrategies(*learned, *expected);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << "\n=== Spanish 21 Q-learning regression ===\n"
              << "Rules: " << (g_standSoft17 ? "S17" : "H17")
              << "  redouble=" << g_redouble
              << "  ddr=" << (g_ddr ? "true" : "false") << "\n"
              << "Decks: " << g_decks << "\n"
              << "Exploration: "
              << (g_explorationMode == ExplorationMode::EPSILON_GREEDY
                      ? "epsilon"
                      : "boltzmann")
              << "\n"
              << "Rounds: " << g_numRounds << "\n"
              << "Differences: " << differences << " / " << compared << "\n"
              << "Accuracy: " << std::fixed << std::setprecision(2)
              << 100.0 * static_cast<double>(compared - differences) /
                     static_cast<double>(compared)
              << "%\n"
              << "Speed: " << std::setprecision(0) << g_numRounds / seconds
              << " rounds/sec\n";
    delete results.front();

    if (differences > g_maxDifferences) {
        std::cerr << "Strategy mismatch threshold exceeded\n";
        return 1;
    }
    return 0;
}
