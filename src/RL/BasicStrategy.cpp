#include "BasicStrategy.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <set>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

BasicStrategy::BasicStrategy() {
    // Initialize shared pointer to empty lookup table
    lookupTable = std::make_shared<std::map<int, std::map<HandType, std::map<int, std::map<int, std::map<int, ActionWithFallback>>>>>>();
}

static HandType parseHandType(const std::string& handTypeStr) {
    if (handTypeStr == "HandType.HARD") {
        return HandType::HARD;
    } else if (handTypeStr == "HandType.SOFT") {
        return HandType::SOFT;
    } else if (handTypeStr == "HandType.PAIR") {
        return HandType::PAIR;
    } else if (handTypeStr == "HandType.ZOMBIE") {
        return HandType::ZOMBIE;
    } else if (handTypeStr == "HandType.BLACKJACK") {
        return HandType::BLACKJACK;
    } else if (handTypeStr == "HandType.AFTER_DOUBLE") {
        return HandType::AFTER_DOUBLE;
    } else if (handTypeStr == "HandType.AFTER_DOUBLE_SOFT") {
        return HandType::AFTER_DOUBLE_SOFT;
    } else {
        std::cerr << "Warning: Unknown HandType string '" << handTypeStr << "', defaulting to HARD" << std::endl;
        return HandType::HARD;
    }
}

// JSON schema: count -> hand_type -> player_sum -> dealer_card -> card_count -> action.
// card_count is always a nested object (even for games that only ever use "2") so the
// lookup table structure is uniform across every game's reference tables.
static bool loadStrategyData(
    const json& jsonData,
    std::shared_ptr<std::map<int, std::map<HandType, std::map<int, std::map<int, std::map<int, ActionWithFallback>>>>>>& lookupTable,
    bool clearExisting = true) {
    if (clearExisting) lookupTable->clear();

    for (auto& [countStr, handTypes] : jsonData.items()) {
        int count = std::stoi(countStr);

        for (auto& [handTypeStr, playerSums] : handTypes.items()) {
            HandType handType = parseHandType(handTypeStr);

            for (auto& [playerSumStr, dealerCards] : playerSums.items()) {
                int playerSum = std::stoi(playerSumStr);

                for (auto& [dealerCardStr, cardCounts] : dealerCards.items()) {
                    int dealerCard = std::stoi(dealerCardStr);

                    for (auto& [cardCountStr, actionStr] : cardCounts.items()) {
                        int cardCount = std::stoi(cardCountStr);
                        std::string actionString = actionStr.get<std::string>();

                        ActionWithFallback action;
                        if (actionString == "H") {
                            action = ActionWithFallback(Action::HIT);
                        } else if (actionString == "S") {
                            action = ActionWithFallback(Action::STAND);
                        } else if (actionString == "Dh") {
                            action = ActionWithFallback(Action::DOUBLE_DOWN, Action::HIT);
                        } else if (actionString == "Ds") {
                            action = ActionWithFallback(Action::DOUBLE_DOWN, Action::STAND);
                        } else if (actionString == "P") {
                            action = ActionWithFallback(Action::SPLIT, Action::HIT);
                        } else if (actionString == "Xh") {
                            action = ActionWithFallback(Action::SURRENDER, Action::HIT);
                        } else if (actionString == "Xs") {
                            action = ActionWithFallback(Action::SURRENDER, Action::STAND);
                        } else {
                            std::cerr << "Warning: Unknown action string '" << actionString << "'" << std::endl;
                            action = ActionWithFallback(Action::HIT);
                        }

                        (*lookupTable)[count][handType][playerSum][dealerCard][cardCount] = action;
                    }
                }
            }
        }
    }
    return true;
}

bool BasicStrategy::loadFromJson(const std::string& filepath) {
    try {
        namespace fs = std::filesystem;
        fs::path full_path = fs::path(PROJECT_ROOT) / "basic_strategy_tables" / "blackjack" / (filepath + ".json");
        std::ifstream file(full_path);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filepath << std::endl;
            return false;
        }
        
        json jsonData;
        file >> jsonData;
        file.close();

        return loadStrategyData(jsonData, lookupTable);
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading strategy: " << e.what() << std::endl;
        return false;
    }
}

bool BasicStrategy::loadFromFile(const std::string& filepath) {
    try {
        namespace fs = std::filesystem;
        fs::path full_path = fs::path(PROJECT_ROOT) / filepath;
        std::ifstream file(full_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << full_path << std::endl;
            return false;
        }

        json jsonData;
        file >> jsonData;
        file.close();
        return loadStrategyData(jsonData, lookupTable);
    } catch (const std::exception& e) {
        std::cerr << "Error loading strategy file: " << e.what() << std::endl;
        return false;
    }
}

bool BasicStrategy::mergeFromFile(const std::string& filepath) {
    try {
        namespace fs = std::filesystem;
        fs::path fullPath = fs::path(PROJECT_ROOT) / filepath;
        std::ifstream file(fullPath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << fullPath << std::endl;
            return false;
        }

        json jsonData;
        file >> jsonData;
        return loadStrategyData(jsonData, lookupTable, /*clearExisting=*/false);
    } catch (const std::exception& e) {
        std::cerr << "Error merging strategy file: " << e.what() << std::endl;
        return false;
    }
}

Action BasicStrategy::getAction(const StateKey& key, const std::vector<Action>& allowedActions) {
    const auto& [count, handType, playerSum, dealerValue, cardCount] = key;

    if(handType == HandType::BLACKJACK)
        return Action::STAND;

    if (allowedActions.size() == 1) {
        return allowedActions[0];
    }

    // Lookup action in table
    try {
        ActionWithFallback actionEntry =
            lookupTable->at(count).at(handType).at(playerSum).at(dealerValue).at(cardCount);

        // Check if the primary action is allowed
        if (std::find(allowedActions.begin(), allowedActions.end(), actionEntry.primary) != allowedActions.end()) {
            return actionEntry.primary;
        }

        // Primary action not allowed, try fallback
        if (std::find(allowedActions.begin(), allowedActions.end(), actionEntry.fallback) != allowedActions.end()) {
            return actionEntry.fallback;
        }

        throw std::logic_error("No intersection between allowed Actions and Strategy Actions");

    } catch (const std::out_of_range&) {
        // Rare post-split edge case:
        // after repeated splits we can end up with hands like 2,2 that are no longer
        // allowed to split, so the state is encoded as HARD 4 instead of PAIR 2.
        // These states are extremely rare and may never be visited during training,
        // so it is reasonable for the strategy table to miss them. In that case we
        // default to HIT, which is a safe fallback and has negligible practical impact.
        if (handType == HandType::HARD && playerSum == 4) {
            return Action::HIT;
        }

        // Same edge case, the soft twin: repeated splitting can also leave a
        // hand like A,A no longer allowed to split, encoded as SOFT 12 instead
        // of PAIR 11 -- reached by the identical rare mechanism as HARD 4
        // above, so it gets the same treatment. A soft 12 can never bust on a
        // hit, and there's no meaningful upside to doubling such a low total,
        // so HIT is the safe, always-correct default here regardless of dealer
        // card -- not a guess, this is standard basic strategy for a hand this
        // weak.
        if (handType == HandType::SOFT && playerSum == 12) {
            return Action::HIT;
        }

        // Spanish 21 rescue/redouble decisions (AFTER_DOUBLE[_SOFT]) may not appear
        // in a hand-transcribed table sourced from a ruleset without that option (no
        // WoO chart exists for every redouble/rescue combination). Standing is
        // always a legal, safe default there.
        if (handType == HandType::AFTER_DOUBLE || handType == HandType::AFTER_DOUBLE_SOFT) {
            return Action::STAND;
        }

        std::ostringstream oss;
        oss << "Error: No strategy entry found for count=" << count
            << ", handType=" << static_cast<int>(handType)
            << ", playerSum=" << playerSum
            << ", dealerValue=" << dealerValue
            << ", cardCount=" << cardCount << std::endl;

        throw std::logic_error(oss.str());
    }
}


// TODO: not belong here
HandType BasicStrategy::stringToHandType(const std::string& handTypeStr) const {
    if (handTypeStr == "HandType.HARD") {
        return HandType::HARD;
    } else if (handTypeStr == "HandType.SOFT") {
        return HandType::SOFT;
    } else if (handTypeStr == "HandType.PAIR") {
        return HandType::PAIR;
    } else if (handTypeStr == "HandType.ZOMBIE") {
        return HandType::ZOMBIE;
    } else if (handTypeStr == "HandType.BLACKJACK") {
        return HandType::BLACKJACK;
    } else if (handTypeStr == "HandType.AFTER_DOUBLE") {
        return HandType::AFTER_DOUBLE;
    } else if (handTypeStr == "HandType.AFTER_DOUBLE_SOFT") {
        return HandType::AFTER_DOUBLE_SOFT;
    } else {
        std::cerr << "Warning: Unknown HandType string '" << handTypeStr << "', defaulting to HARD" << std::endl;
        return HandType::HARD;
    }
}

std::unique_ptr<Strategy> BasicStrategy::clone() const {
    auto cloned = std::make_unique<BasicStrategy>();
    // Share the lookup table pointer instead of copying (it's read-only during gameplay)
    cloned->lookupTable = this->lookupTable;
    return cloned;
}

void BasicStrategy::printTo(std::ostream& os) const {
    os << *this;
}

void BasicStrategy::setAction(int count, HandType handType, unsigned int playerSum, unsigned int dealerHand,
                              ActionWithFallback action, unsigned int cardCount) {
    // Initialize lookup table if it doesn't exist
    if (!lookupTable) {
        lookupTable = std::make_shared<std::map<int, std::map<HandType, std::map<int, std::map<int, std::map<int, ActionWithFallback>>>>>>();
    }

    // Set the action in the lookup table
    (*lookupTable)[count][handType][playerSum][dealerHand][cardCount] = action;
}

ActionWithFallback BasicStrategy::getActionFromTable(int count, HandType handType, int playerSum, int dealerCard,
                                                      int cardCount) const {
    // AFTER_DOUBLE[_SOFT]'s safe default is STAND (mirrors getAction()'s fallback) --
    // a HIT default is meaningless there since HIT is never a legal AFTER_DOUBLE
    // action, and this handType is often genuinely absent from a table sourced
    // from a ruleset without redouble/rescue (see getAction()'s catch block).
    const ActionWithFallback missingDefault =
        (handType == HandType::AFTER_DOUBLE || handType == HandType::AFTER_DOUBLE_SOFT)
        ? ActionWithFallback(Action::STAND)
        : ActionWithFallback(Action::HIT, Action::HIT);
    if (!lookupTable) {
        return missingDefault;
    }
    try {
        return lookupTable->at(count).at(handType).at(playerSum).at(dealerCard).at(cardCount);
    } catch (const std::out_of_range&) {
        return missingDefault;
    }
}

bool BasicStrategy::hasEntry(int count, HandType handType, int playerSum, int dealerCard,
                             int cardCount) const {
    if (!lookupTable) return false;
    try {
        lookupTable->at(count).at(handType).at(playerSum).at(dealerCard).at(cardCount);
        return true;
    } catch (const std::out_of_range&) {
        return false;
    }
}

std::pair<int, int> BasicStrategy::getCountRange() const {
    if (!lookupTable || lookupTable->empty()) {
        return {0, 0};
    }
    return {lookupTable->begin()->first, lookupTable->rbegin()->first};
}

// Helper function to convert action to string for display
static std::string actionToString(const ActionWithFallback& action) {
    if (action.primary == Action::HIT) {
        return "H";
    } else if (action.primary == Action::STAND) {
        return "S";
    } else if (action.primary == Action::DOUBLE_DOWN) {
        if (action.fallback == Action::HIT) {
            return "Dh";
        } else if (action.fallback == Action::STAND) {
            return "Ds";
        } else {
            return "D";
        }
    } else if (action.primary == Action::SPLIT) {
        return "P";
    } else if (action.primary == Action::SURRENDER) {
        if (action.fallback == Action::HIT) {
            return "Xh";
        } else if (action.fallback == Action::STAND) {
            return "Xs";
        } else {
            return "X";
        }
    }
    return "?";
}

// Helper function to convert HandType to string for display
static std::string handTypeToString(HandType handType) {
    switch (handType) {
        case HandType::HARD:         return "Hard";
        case HandType::SOFT:         return "Soft";
        case HandType::PAIR:         return "Pair";
        case HandType::ZOMBIE:       return "One card";
        case HandType::BLACKJACK:    return "BlackJack";
        case HandType::AFTER_DOUBLE:      return "After Double";
        case HandType::AFTER_DOUBLE_SOFT: return "After Double (Soft)";
        default:                          return "Unknown";
    }
}

// Serialize lookup table to JSON (same format as basic_strategy_tables/blackjack/*.json)
json BasicStrategy::toJson() const {
    json result;
    if (!lookupTable) return result;

    for (const auto& [count, handTypes] : *lookupTable) {
        std::string countStr = std::to_string(count);
        for (const auto& [handType, playerSums] : handTypes) {
            if (handType == HandType::BLACKJACK) continue;
            std::string handTypeStr;
            switch (handType) {
                case HandType::HARD: handTypeStr = "HandType.HARD"; break;
                case HandType::SOFT: handTypeStr = "HandType.SOFT"; break;
                case HandType::PAIR: handTypeStr = "HandType.PAIR"; break;
                case HandType::ZOMBIE: handTypeStr = "HandType.ZOMBIE"; break;
                case HandType::AFTER_DOUBLE: handTypeStr = "HandType.AFTER_DOUBLE"; break;
                case HandType::AFTER_DOUBLE_SOFT: handTypeStr = "HandType.AFTER_DOUBLE_SOFT"; break;
                default: continue;
            }
            for (const auto& [playerSum, dealerCards] : playerSums) {
                std::string playerSumStr = std::to_string(playerSum);
                for (const auto& [dealerCard, cardCounts] : dealerCards) {
                    std::string dealerCardStr = std::to_string(dealerCard);
                    for (const auto& [cardCount, action] : cardCounts) {
                        result[countStr][handTypeStr][playerSumStr][dealerCardStr][std::to_string(cardCount)] =
                            actionToString(action);
                    }
                }
            }
        }
    }
    return result;
}

bool BasicStrategy::saveToJson(const std::string& filepath) const {
    try {
        namespace fs = std::filesystem;
        fs::path full_path = fs::path(PROJECT_ROOT) / filepath;
        fs::create_directories(full_path.parent_path());
        std::ofstream file(full_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << full_path << std::endl;
            return false;
        }
        file << toJson().dump(2);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving strategy: " << e.what() << std::endl;
        return false;
    }
}

// Groups the cardCount->action entries for one (playerSum, dealerCard) cell into
// compact ranges describing where the action deviates from the base (lowest)
// card count -- e.g. "4+: H" or "4-5: H". Adjacent card counts sharing the same
// non-base action are merged into one range; maxCardCount (the capped "N or
// more" bucket) is rendered as "N+" rather than "N-<max>".
static std::vector<std::string> describeCardCountDeviations(
        const std::map<int, ActionWithFallback>& byCardCount, int baseCardCount, int maxCardCount) {
    const std::string baseStr = actionToString(byCardCount.at(baseCardCount));
    std::vector<std::string> notes;
    int rangeStart = -1, rangeEnd = -1;
    std::string rangeAction;

    auto flush = [&]() {
        if (rangeStart == -1) return;
        std::string label = (rangeStart == rangeEnd)
            ? std::to_string(rangeStart)
            : (rangeEnd == maxCardCount ? std::to_string(rangeStart) + "+"
                                        : std::to_string(rangeStart) + "-" + std::to_string(rangeEnd));
        notes.push_back(label + " cards: " + rangeAction);
        rangeStart = -1;
    };

    for (const auto& [cardCount, action] : byCardCount) {
        if (cardCount == baseCardCount) continue;
        const std::string actionStr = actionToString(action);
        if (actionStr == baseStr) {
            flush();
            continue;
        }
        if (rangeStart != -1 && actionStr == rangeAction && cardCount == rangeEnd + 1) {
            rangeEnd = cardCount;
        } else {
            flush();
            rangeStart = rangeEnd = cardCount;
            rangeAction = actionStr;
        }
    }
    flush();
    return notes;
}

// Operator<< for printing the strategy table. Mirrors wizardofodds.com's own chart
// layout: one table per hand type (not one per card count -- with the count-embedded
// deviation feature planned, count x handType x cardCount would otherwise multiply
// into hundreds of near-duplicate tables). The table shows the base (lowest) card
// count's action; cells where a higher card count changes the decision are listed
// as compact notes below the table instead of duplicating the whole grid.
std::ostream& operator<<(std::ostream& os, const BasicStrategy& strategy) {
    if (!strategy.lookupTable || strategy.lookupTable->empty()) {
        os << "Strategy table is empty or not loaded.\n";
        return os;
    }

    // Iterate through counts
    for (const auto& [count, handTypes] : *strategy.lookupTable) {
        os << "\n========== Count: " << count << " ==========\n";

        // Iterate through hand types.
        for (const auto& [handType, playerSums] : handTypes) {
            os << "\n" << handTypeToString(handType) << "\n";

            // Find all unique dealer cards, player sums, and card counts
            std::set<int> dealerCards;
            std::set<int> playerSumValues;
            std::set<int> cardCounts;

            for (const auto& [playerSum, dealerMap] : playerSums) {
                playerSumValues.insert(playerSum);
                for (const auto& [dealerCard, cardCountMap] : dealerMap) {
                    dealerCards.insert(dealerCard);
                    for (const auto& [cardCount, action] : cardCountMap) {
                        cardCounts.insert(cardCount);
                    }
                }
            }
            if (dealerCards.empty() || playerSumValues.empty() || cardCounts.empty()) continue;
            const int baseCardCount = *cardCounts.begin();
            const int maxCardCount = *cardCounts.rbegin();

            auto dealerLabel = [](int dealerCard) {
                return dealerCard == 11 ? "A" : std::to_string(dealerCard);
            };

            // Print header row (dealer cards)
            os << "    ";
            for (int dealerCard : dealerCards) {
                os << std::setw(4) << dealerLabel(dealerCard);
            }
            os << "\n";

            // Print each row (player sum), using the base card count's action, and
            // collect any higher-card-count deviations to print below the grid.
            std::vector<std::string> deviationNotes;
            for (int playerSum : playerSumValues) {
                const std::string playerLabel =
                    handType == HandType::ZOMBIE && playerSum == 11
                        ? "A"
                        : std::to_string(playerSum);
                os << std::setw(3) << playerLabel << " ";

                for (int dealerCard : dealerCards) {
                    auto dealerIt = playerSums.at(playerSum).find(dealerCard);
                    if (dealerIt == playerSums.at(playerSum).end() ||
                        !dealerIt->second.count(baseCardCount)) {
                        os << std::setw(4) << "-";
                        continue;
                    }
                    const auto& byCardCount = dealerIt->second;
                    os << std::setw(4) << actionToString(byCardCount.at(baseCardCount));

                    if (byCardCount.size() > 1) {
                        for (const std::string& note :
                             describeCardCountDeviations(byCardCount, baseCardCount, maxCardCount)) {
                            deviationNotes.push_back(
                                playerLabel + " vs " + dealerLabel(dealerCard) + ": " + note);
                        }
                    }
                }
                os << "\n";
            }

            if (!deviationNotes.empty()) {
                os << "  Card-count deviations (base is " << baseCardCount << " cards):\n";
                for (const std::string& note : deviationNotes) {
                    os << "    " << note << "\n";
                }
            }
        }
    }

    return os;
}
