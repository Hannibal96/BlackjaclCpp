#include "BasicStrategy.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <set>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

BasicStrategy::BasicStrategy() {
    // Initialize shared pointer to empty lookup table
    lookupTable = std::make_shared<std::map<int, std::map<HandType, std::map<int, std::map<int, ActionWithFallback>>>>>();
}

bool BasicStrategy::loadFromJson(const std::string& filepath) {
    try {
        namespace fs = std::filesystem;
        fs::path full_path = fs::path(PROJECT_ROOT) / "basic_strategy_tables" / (filepath + ".json");
        std::ifstream file(full_path);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filepath << std::endl;
            return false;
        }
        
        json jsonData;
        file >> jsonData;
        file.close();
        
        // Clear existing lookup table
        lookupTable->clear();
        
        // Parse the JSON structure: count -> hand_type -> player_sum -> dealer_card -> action
        for (auto& [countStr, handTypes] : jsonData.items()) {
            int count = std::stoi(countStr);
            
            for (auto& [handTypeStr, playerSums] : handTypes.items()) {
                HandType handType = stringToHandType(handTypeStr);
                
                for (auto& [playerSumStr, dealerCards] : playerSums.items()) {
                    int playerSum = std::stoi(playerSumStr);
                    
                    for (auto& [dealerCardStr, actionStr] : dealerCards.items()) {
                        int dealerCard = std::stoi(dealerCardStr);
                        std::string actionString = actionStr.get<std::string>();
                        
                        // Convert action string to ActionWithFallback
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
                        
                        (*lookupTable)[count][handType][playerSum][dealerCard] = action;
                    }
                }
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading strategy: " << e.what() << std::endl;
        return false;
    }
}

Action BasicStrategy::getAction(const State& state) {
    // Get hand type
    bool splitAllowed = std::find(state.allowedActions.begin(), state.allowedActions.end(), Action::SPLIT) != state.allowedActions.end();
    HandType handType = state.playerHand.getHandType(splitAllowed);
    if (handType == HandType::ZOMBIE) {
        return Action::HIT;
    }
    if(state.allowedActions.size() == 1){
        return state.allowedActions[0];
    }
    
    // Get player sum - for pairs, use the value of one card, not the total
    int playerSum;
    if (handType == HandType::PAIR) {
        // For pairs, use the value of one card (not the total), this is for the case of 66 and AA which can cause confustions 
        playerSum = state.playerHand[0].getValue();
    } else {
        playerSum = state.playerHand.getValue();
    }
    
    // Get dealer card value
    int dealerValue = state.dealerCard.getValue();
    
    // Use count from state
    int count = state.count;
    
    // Lookup action in table
    try {
        ActionWithFallback actionEntry = lookupTable->at(count).at(handType).at(playerSum).at(dealerValue);
        
        // Check if the primary action is allowed
        const auto& allowed = state.allowedActions;
        if (std::find(allowed.begin(), allowed.end(), actionEntry.primary) != allowed.end()) {
            return actionEntry.primary;
        }
        
        // Primary action not allowed, try fallback
        if (std::find(allowed.begin(), allowed.end(), actionEntry.fallback) != allowed.end()) {
            return actionEntry.fallback;
        }
        
        throw std::logic_error("No intersection between allowed Actions and Strategy Actions");
        
    } catch (const std::out_of_range& e) {
        std::ostringstream oss;
        oss << "Error: No strategy entry found for count=" << count 
                  << ", handType=" << static_cast<int>(handType)
                  << ", playerSum=" << playerSum 
                  << ", dealerValue=" << dealerValue << std::endl;
        
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

void BasicStrategy::setAction(int count, HandType handType, unsigned int playerSum, unsigned int dealerHand, ActionWithFallback action) {
    // Initialize lookup table if it doesn't exist
    if (!lookupTable) {
        lookupTable = std::make_shared<std::map<int, std::map<HandType, std::map<int, std::map<int, ActionWithFallback>>>>>();
    }
    
    // Set the action in the lookup table
    (*lookupTable)[count][handType][playerSum][dealerHand] = action;
}

ActionWithFallback BasicStrategy::getActionFromTable(int count, HandType handType, int playerSum, int dealerCard) const {
    if (!lookupTable) {
        return {Action::HIT, Action::HIT};
    }
    try {
        return lookupTable->at(count).at(handType).at(playerSum).at(dealerCard);
    } catch (const std::out_of_range&) {
        return {Action::HIT, Action::HIT};
    }
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
        case HandType::HARD:      return "Hard";
        case HandType::SOFT:      return "Soft";
        case HandType::PAIR:      return "Pair";
        case HandType::BLACKJACK: return "BlackJack";
        default:                  return "Unknown";
    }
}

// Operator<< for printing the strategy table
std::ostream& operator<<(std::ostream& os, const BasicStrategy& strategy) {
    if (!strategy.lookupTable || strategy.lookupTable->empty()) {
        os << "Strategy table is empty or not loaded.\n";
        return os;
    }
    
    // Iterate through counts
    for (const auto& [count, handTypes] : *strategy.lookupTable) {
        os << "\n========== Count: " << count << " ==========\n";
        
        // Iterate through hand types (HARD, SOFT, PAIR)
        for (const auto& [handType, playerSums] : handTypes) {
            if(handType == HandType::ZOMBIE) {
                continue; // Skip ZOMBIE hand type
            }
            os << "\n" << handTypeToString(handType) << "\n";
            
            // Find all unique dealer cards and player sums
            std::set<int> dealerCards;
            std::set<int> playerSumValues;
            
            for (const auto& [playerSum, dealerMap] : playerSums) {
                playerSumValues.insert(playerSum);
                for (const auto& [dealerCard, action] : dealerMap) {
                    dealerCards.insert(dealerCard);
                }
            }
            
            // Print header row (dealer cards)
            os << "    ";
            for (int dealerCard : dealerCards) {
                os << std::setw(4) << dealerCard;
            }
            os << "\n";
            
            // Print each row (player sum)
            for (int playerSum : playerSumValues) {
                os << std::setw(3) << playerSum << " ";
                
                for (int dealerCard : dealerCards) {
                    auto dealerIt = playerSums.at(playerSum).find(dealerCard);
                    if (dealerIt != playerSums.at(playerSum).end()) {
                        os << std::setw(4) << actionToString(dealerIt->second);
                    } else {
                        os << std::setw(4) << "-";
                    }
                }
                os << "\n";
            }
        }
    }
    
    return os;
}

