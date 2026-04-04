#pragma once
#include "Strategy.h"
#include "DecayingParameter.h"
#include "BasicStrategy.h"
#include <map>
#include <tuple>
#include <memory>
#include <random>
#include <algorithm>
#include <cmath>

enum class ExplorationMode { EPSILON_GREEDY, BOLTZMANN };

// Q-Learning strategy for reinforcement learning
class QLearningStrategy : public Strategy {
private:
    // Q-table: maps (state_representation, action) to Q-value
    using StateKey = std::tuple<int, HandType, unsigned int, unsigned int>;  // (count, hand_type, player_sum, dealer_hand)
    using QTableKey = std::pair<StateKey, Action>;
    std::map<QTableKey, double> qTable;

    // Per-state learning parameters
    std::map<StateKey, std::unique_ptr<DecayingParameter>> alphaMap;       // Per-state learning rate
    std::map<StateKey, std::unique_ptr<DecayingParameter>> explorationMap; // Per-state epsilon (greedy) or temperature (Boltzmann)

    // Template parameters for creating new state entries
    std::unique_ptr<DecayingParameter> alphaTemplate;
    std::unique_ptr<DecayingParameter> explorationTemplate;

    double gamma;             // Discount factor (global)
    ExplorationMode explorationMode;

    // RNG — used by both exploration strategies
    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<double> dist;

    // Helper method to convert State to a simplified key for Q-table
    StateKey stateToKey(const State& state) const {
        int count = state.count;
        HandType handType = state.playerHand.getHandType();
        unsigned int dealerHand = state.dealerCard.getValue();
        unsigned int playerSum;
        if(handType == HandType::PAIR) {
            // For pairs, use the value of one card (not the total), this is for the case of 66 and AA which can cause confustions 
            playerSum = state.playerHand[0].getValue();
        } else {
            playerSum = state.playerHand.getValue();
        }
        return std::make_tuple(count, handType, playerSum, dealerHand);
    }
    
    // Get or create a decaying parameter for a state from the given map using the template
    DecayingParameter& getOrCreateParam(const StateKey& key, 
                                        std::map<StateKey, std::unique_ptr<DecayingParameter>>& paramMap,
                                        const std::unique_ptr<DecayingParameter>& paramTemplate) {
        auto it = paramMap.find(key);
        if (it == paramMap.end()) {
            paramMap[key] = paramTemplate->clone();
            return *paramMap[key];
        }
        return *it->second;
    }
    
    // Get Q-value for a state-action pair (returns 0 if not in table)
    double getQValue(const State& state, Action action) const {
        StateKey key = stateToKey(state);
        auto it = qTable.find(std::make_pair(key, action));
        return (it != qTable.end()) ? it->second : 0.0;
    }

    
    // Get the maximum Q-value for a given state across all allowed actions
    double getMaxQValue(const State& state) const {
        double maxQ = -std::numeric_limits<double>::infinity();
        bool foundAction = false;
        
        for (const Action& action : state.allowedActions) {
            double q = getQValue(state, action);
            if (!foundAction || q > maxQ) {
                maxQ = q;
                foundAction = true;
            }
        }
        
        return foundAction ? maxQ : 0.0;
    }
    
    // Get the greedy action (action with highest Q-value)
    Action getGreedyAction(const State& state) const {
        Action bestAction = state.allowedActions[0];
        double maxQ = getQValue(state, bestAction);
        
        for (const Action& action : state.allowedActions) {
            double q = getQValue(state, action);
            if (q > maxQ) {
                maxQ = q;
                bestAction = action;
            }
        }
        
        return bestAction;
    }
    
public:
    __attribute__((noinline)) double getQValueDebug(const HandType handType, int playerSum, int dealerHand, Action action) const;

    // Constructor — exploration_param is epsilon for EPSILON_GREEDY, temperature for BOLTZMANN
    QLearningStrategy(std::unique_ptr<DecayingParameter> alpha_param,
                      std::unique_ptr<DecayingParameter> exploration_param,
                      double discount_factor = 1.0,
                      ExplorationMode mode = ExplorationMode::EPSILON_GREEDY)
        : alphaTemplate(std::move(alpha_param)), explorationTemplate(std::move(exploration_param)),
          gamma(discount_factor), explorationMode(mode),
          rng(std::random_device{}()), dist(0.0, 1.0) {}

    Action getAction(const State& state) override {
        StateKey key = stateToKey(state);
        DecayingParameter& stateExploration = getOrCreateParam(key, explorationMap, explorationTemplate);

        if (explorationMode == ExplorationMode::EPSILON_GREEDY) {
            // Epsilon-greedy: explore with probability epsilon, exploit otherwise
            if (dist(rng) < stateExploration.getValue()) {
                stateExploration.updateValue();
                std::uniform_int_distribution<size_t> actionDist(0, state.allowedActions.size() - 1);
                return state.allowedActions[actionDist(rng)];
            } else {
                return getGreedyAction(state);
            }
        } else {
            // Boltzmann (softmax): P(a|s) = exp(Q(s,a)/τ) / Σ exp(Q(s,b)/τ)
            double temperature = stateExploration.getValue();

            std::vector<double> qValues;
            qValues.reserve(state.allowedActions.size());
            for (const Action& action : state.allowedActions) {
                qValues.push_back(getQValue(state, action));
            }

            // log-sum-exp trick for numerical stability
            double maxQ = *std::max_element(qValues.begin(), qValues.end());
            std::vector<double> weights;
            weights.reserve(qValues.size());
            double sumExp = 0.0;
            for (double q : qValues) {
                double w = std::exp((q - maxQ) / temperature);
                weights.push_back(w);
                sumExp += w;
            }

            double r = dist(rng) * sumExp;
            double cumulative = 0.0;
            for (size_t i = 0; i < state.allowedActions.size(); ++i) {
                cumulative += weights[i];
                if (r <= cumulative) {
                    stateExploration.updateValue();
                    return state.allowedActions[i];
                }
            }
            stateExploration.updateValue();
            return state.allowedActions.back();  // fallback for floating-point edge case
        }
    }
    
    // Update Q-table using Q-learning update rule
    // Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
    void updateTable(const State& state, Action action, double reward, const State& nextState) override {
        StateKey key = stateToKey(state);
        DecayingParameter& stateAlpha = getOrCreateParam(key, alphaMap, alphaTemplate);
        
        // Get current Q-value
        double currentQ = getQValue(state, action);
        
        // Get maximum Q-value for next state
        double maxNextQ = getMaxQValue(nextState);
        
        // Special handling for SPLIT: we play two hands, so expected value doubles
        if (action == Action::SPLIT) {
            maxNextQ *= 2.0;
        }
        
        // Q-learning update with per-state alpha
        double newQ = currentQ + stateAlpha.getValue() * (reward + gamma * maxNextQ - currentQ);
        
        // Update Q-table
        qTable[std::make_pair(key, action)] = newQ;
        stateAlpha.updateValue();  // Decay alpha for this state
    }
    
    // Update alpha for a specific state (decay learning rate)
    void updateAlpha(const State& state) {
        StateKey key = stateToKey(state);
        getOrCreateParam(key, alphaMap, alphaTemplate).updateValue();
    }
    
    // Update exploration param for a specific state (decay epsilon or temperature)
    void updateExploration(const State& state) {
        StateKey key = stateToKey(state);
        getOrCreateParam(key, explorationMap, explorationTemplate).updateValue();
    }

    // Get alpha value for a specific state
    double getAlpha(const State& state) {
        StateKey key = stateToKey(state);
        return getOrCreateParam(key, alphaMap, alphaTemplate).getValue();
    }

    // Get exploration param value for a specific state (epsilon or temperature)
    double getExploration(const State& state) {
        StateKey key = stateToKey(state);
        return getOrCreateParam(key, explorationMap, explorationTemplate).getValue();
    }

    // Get template alpha value (initial value for new states)
    double getTemplateAlpha() const {
        return alphaTemplate->getValue();
    }

    // Get template exploration value (initial value for new states)
    double getTemplateExploration() const {
        return explorationTemplate->getValue();
    }

    ExplorationMode getExplorationMode() const { return explorationMode; }
    
    // Get Q-table size (for debugging/monitoring)
    size_t getTableSize() const {
        return qTable.size();
    }
    
    // Clone method for Strategy interface
    std::unique_ptr<Strategy> clone() const override {
        auto cloned = std::make_unique<QLearningStrategy>(
            alphaTemplate->clone(), explorationTemplate->clone(), gamma, explorationMode);

        cloned->qTable = this->qTable;

        for (const auto& [key, param] : alphaMap) {
            cloned->alphaMap[key] = param->clone();
        }
        for (const auto& [key, param] : explorationMap) {
            cloned->explorationMap[key] = param->clone();
        }

        return cloned;
    }
    
    // Convert Q-learning strategy to BasicStrategy by extracting best actions
    std::unique_ptr<BasicStrategy> toBasicStrategy() const {
        auto basicStrategy = std::make_unique<BasicStrategy>();
        
        // Group Q-table entries by state to find best and second-best actions
        std::map<StateKey, std::vector<std::pair<Action, double>>> stateActions;
        
        for (const auto& [key, qValue] : qTable) {
            const auto& [stateKey, action] = key;
            stateActions[stateKey].push_back({action, qValue});
        }
        
        // For each state, find best action and appropriate fallback
        for (auto& [stateKey, actions] : stateActions) {
            // Sort actions by Q-value (descending)
            std::sort(actions.begin(), actions.end(), 
                [](const auto& a, const auto& b) { return a.second > b.second; });
            
            if (actions.empty()) continue;
            
            Action bestAction = actions[0].first;
            Action fallbackAction = bestAction;
            
            // Determine fallback based on best action type
            if (bestAction == Action::SURRENDER || bestAction == Action::DOUBLE_DOWN || bestAction == Action::SPLIT) {
                // Find second-best action that is HIT or STAND
                for (size_t i = 1; i < actions.size(); ++i) {
                    Action candidateAction = actions[i].first;
                    if (candidateAction == Action::HIT || candidateAction == Action::STAND) {
                        fallbackAction = candidateAction;
                        break;
                    }
                }
                
                // If no HIT/STAND found, default based on action type
                if (fallbackAction == bestAction) {
                    if (bestAction == Action::DOUBLE_DOWN || bestAction == Action::SURRENDER) {
                        fallbackAction = Action::HIT;  // Default fallback for DOUBLE/SURRENDER
                    } else if (bestAction == Action::SPLIT) {
                        fallbackAction = Action::HIT;  // Default fallback for SPLIT
                    }
                }
            }
            
            const auto& [count, handType, playerSum, dealerHand] = stateKey;
            basicStrategy->setAction(count, handType, playerSum, dealerHand, 
                ActionWithFallback(bestAction, fallbackAction));
        }
        
        return basicStrategy;
    }

    // Averaging operators for combining Q-tables from parallel simulations
    Strategy& operator+=(const Strategy& other) override {
        // Try to cast to QLearningStrategy
        const QLearningStrategy* otherQL = dynamic_cast<const QLearningStrategy*>(&other);
        if (otherQL) {
            // Sum each entry in the Q-table
            for (const auto& [key, value] : otherQL->qTable) {
                qTable[key] += value;
            }
        }
        return *this;
    }
    
    Strategy& operator*=(double factor) override {
        // Multiply each entry in the Q-table by the factor
        for (auto& [key, value] : qTable) {
            value *= factor;
        }
        return *this;
    }

    // Friend function for printing the Q-learning strategy table
    friend std::ostream& operator<<(std::ostream& os, const QLearningStrategy& strategy) {
        // Convert Q-table to BasicStrategy format and print it
        auto basicStrategy = strategy.toBasicStrategy();
        os << *basicStrategy;
        return os;
    }

};
