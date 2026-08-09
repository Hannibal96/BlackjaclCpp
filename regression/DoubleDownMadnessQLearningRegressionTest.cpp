#include "Game/DoubleDownMadnessRules.h"
#include "Game/Player.h"
#include "RL/DecayingParameter.h"
#include "RL/DoubleDownMadnessStrategy.h"
#include "RL/QLearningStrategy.h"
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
int g_decks = 6;
int g_version = 1;
int g_maxDifferences = 10;
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

DoubleDownMadnessVersion versionFromInt(int value) {
    if (value < 1 || value > 3)
        throw std::invalid_argument("Version must be 1, 2, or 3");
    return static_cast<DoubleDownMadnessVersion>(value);
}

const char* handTypeName(HandType type) {
    switch (type) {
        case HandType::HARD: return "HARD";
        case HandType::SOFT: return "SOFT";
        case HandType::PAIR: return "PAIR";
        case HandType::ZOMBIE: return "ONE_CARD";
        case HandType::BLACKJACK: return "BLACKJACK";
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

void verifyStateEncoding(const Player& player) {
    const Card dealerSix(Rank::SIX, Suit::CLUBS);

    Hand oneCardAce(1.0);
    oneCardAce.addCard(Card(Rank::ACE, Suit::SPADES));
    const StateKey aceKey = player.stateToKey(
        State(oneCardAce, dealerSix, {Action::HIT, Action::DOUBLE_DOWN}));

    Hand hardEleven(1.0);
    hardEleven.addCard(Card(Rank::FIVE, Suit::HEARTS));
    hardEleven.addCard(Card(Rank::SIX, Suit::DIAMONDS));
    const StateKey hardElevenKey = player.stateToKey(
        State(hardEleven, dealerSix,
              {Action::HIT, Action::STAND, Action::DOUBLE_DOWN}));

    Hand softTwelve(1.0);
    softTwelve.addCard(Card(Rank::ACE, Suit::SPADES));
    softTwelve.addCard(Card(Rank::ACE, Suit::DIAMONDS));
    const StateKey softTwelveKey = player.stateToKey(
        State(softTwelve, dealerSix,
              {Action::HIT, Action::STAND, Action::DOUBLE_DOWN}));

    Hand blackjack(1.0);
    blackjack.addCard(Card(Rank::ACE, Suit::HEARTS));
    blackjack.addCard(Card(Rank::KING, Suit::HEARTS));
    const StateKey blackjackKey =
        player.stateToKey(State(blackjack, dealerSix, {}));

    Hand ordinaryTwentyOne(1.0);
    ordinaryTwentyOne.addCard(Card(Rank::SEVEN, Suit::CLUBS));
    ordinaryTwentyOne.addCard(Card(Rank::SEVEN, Suit::DIAMONDS));
    ordinaryTwentyOne.addCard(Card(Rank::SEVEN, Suit::SPADES));
    const StateKey ordinaryTwentyOneKey =
        player.stateToKey(State(ordinaryTwentyOne, dealerSix, {}));

    if (aceKey == hardElevenKey ||
        std::get<1>(aceKey) != HandType::ZOMBIE ||
        std::get<2>(aceKey) != 11 ||
        std::get<1>(hardElevenKey) != HandType::HARD ||
        std::get<2>(hardElevenKey) != 11) {
        throw std::logic_error(
            "State encoding aliases one-card Ace with hard 11");
    }
    if (blackjackKey == ordinaryTwentyOneKey ||
        std::get<1>(blackjackKey) != HandType::BLACKJACK ||
        std::get<2>(blackjackKey) != 21 ||
        std::get<1>(ordinaryTwentyOneKey) != HandType::HARD ||
        std::get<2>(ordinaryTwentyOneKey) != 21) {
        throw std::logic_error(
            "State encoding aliases blackjack with ordinary 21");
    }
    if (std::get<1>(softTwelveKey) != HandType::SOFT ||
        std::get<2>(softTwelveKey) != 12) {
        throw std::logic_error("A,A is not encoded as actionable soft 12");
    }
}

std::pair<int, int> compareStrategies(const BasicStrategy& learned,
                                      const BasicStrategy& expected) {
    int differences = 0;
    int compared = 0;
    auto compareState = [&](HandType type, int total, int dealer) {
        const Action learnedAction =
            learned.getActionFromTable(0, type, total, dealer).primary;
        const Action expectedAction =
            expected.getActionFromTable(0, type, total, dealer).primary;
        ++compared;
        if (learnedAction != expectedAction) {
            ++differences;
            std::cout << "Mismatch handType=" << handTypeName(type)
                      << " total=" << total << " dealer=" << dealer
                      << " learned=" << actionName(learnedAction)
                      << " expected=" << actionName(expectedAction) << "\n";
        }
    };

    for (int total = 2; total <= 11; ++total)
        for (int dealer = 2; dealer <= 11; ++dealer)
            compareState(HandType::ZOMBIE, total, dealer);
    for (int total = 4; total <= 20; ++total)
        for (int dealer = 2; dealer <= 11; ++dealer)
            compareState(HandType::HARD, total, dealer);
    for (int total = 12; total <= 20; ++total)
        for (int dealer = 2; dealer <= 11; ++dealer)
            compareState(HandType::SOFT, total, dealer);
    return {differences, compared};
}

void printHelp(const char* program) {
    std::cout
        << "Usage: " << program << " [OPTIONS]\n\n"
        << "Train a Double Down Madness Q-learning policy and compare it with\n"
        << "the known basic strategy.\n\n"
        << "  --num-rounds <N>       Training rounds (default: 1000000000)\n"
        << "  --num-threads <N>      Threads (default: 10)\n"
        << "  --penetration <P>      Penetration (default: 75)\n"
        << "  --decks <N>            Number of decks (default: 6)\n"
        << "  --version <1|2|3>      Paytable version (default: 1)\n"
        << "  --max-differences <N>  Allowed mismatches (default: 10)\n"
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
        else if (arg == "--version" && i + 1 < argc)
            g_version = std::stoi(argv[++i]);
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
    const DoubleDownMadnessVersion version = versionFromInt(g_version);
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
    verifyStateEncoding(*player);

    DoubleDownMadnessRules rules(
        version, /*standS17=*/false, g_decks, g_penetration);
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
    auto expected = loadDoubleDownMadnessBasicStrategy(version);
    const auto [differences, compared] = compareStrategies(*learned, *expected);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::cout << "\n=== Double Down Madness Q-learning regression ===\n"
              << "Version: " << g_version << "\n"
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
