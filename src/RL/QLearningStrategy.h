#pragma once
#include "Strategy.h"
#include "DecayingParameter.h"
#include "BasicStrategy.h"
#include <map>
#include <tuple>
#include <memory>
#include <random>
#include <algorithm>

// Q-Learning strategy for reinforcement learning
class QLearningStrategy : public Strategy {
private:
    // Q-table: maps (state_representation, action) to Q-value
    using StateKey = std::tuple<int, HandType, unsigned int, unsigned int>;  // (count, hand_type, player_sum, dealer_hand)
    using QTableKey = std::pair<StateKey, Action>;
    std::map<QTableKey, double> qTable;
    
    // Learning parameters
    std::unique_ptr<DecayingParameter> alpha;  // Learning rate (can decay)
    double gamma;           // Discount factor
    std::unique_ptr<DecayingParameter> epsilon;  // Exploration rate (can decay)
    
    // Random number generator for epsilon-greedy exploration
    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<double> dist;
    
    // Helper method to convert State to a simplified key for Q-table
    StateKey stateToKey(const State& state) const {
        int count = state.count;
        HandType handType = state.playerHand.getHandType();
        unsigned int playerSum = state.playerHand.getValue();
        unsigned int dealerHand = state.dealerCard.getValue();
        return std::make_tuple(count, handType, playerSum, dealerHand);
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
    // Constructor with learning rate, discount factor, and epsilon (exploration rate)
    QLearningStrategy(std::unique_ptr<DecayingParameter> alpha_param, 
                     std::unique_ptr<DecayingParameter> epsilon_param, double discount_factor=1.0)
        : alpha(std::move(alpha_param)), gamma(discount_factor), 
          epsilon(std::move(epsilon_param)),
          rng(std::random_device{}()), dist(0.0, 1.0) {}
    
    // Epsilon-greedy action selection
    Action getAction(const State& state) override {
        // Epsilon-greedy: explore with probability epsilon, exploit otherwise
        if (dist(rng) < epsilon->getValue()) {
            // Explore: choose random action from allowed actions
            std::uniform_int_distribution<size_t> actionDist(0, state.allowedActions.size() - 1);
            return state.allowedActions[actionDist(rng)];
        } else {
            // Exploit: choose greedy action
            return getGreedyAction(state);
        }
    }
    
    // Update Q-table using Q-learning update rule
    // Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
    void updateTable(const State& state, Action action, double reward, const State& nextState) {
        // Get current Q-value
        double currentQ = getQValue(state, action);
        
        // Get maximum Q-value for next state
        double maxNextQ = getMaxQValue(nextState);
        
        // Q-learning update
        double newQ = currentQ + alpha->getValue() * (reward + gamma * maxNextQ - currentQ);
        
        // Update Q-table
        StateKey key = stateToKey(state);
        qTable[std::make_pair(key, action)] = newQ;
    }
    
    // Update alpha (decay learning rate)
    void updateAlpha() {
        alpha->updateValue();
    }
    
    // Update epsilon (decay exploration rate)
    void updateEpsilon() {
        epsilon->updateValue();
    }
    
    // Get current alpha value
    double getAlpha() const {
        return alpha->getValue();
    }
    
    // Get current epsilon value
    double getEpsilon() const {
        return epsilon->getValue();
    }
    
    // Get Q-table size (for debugging/monitoring)
    size_t getTableSize() const {
        return qTable.size();
    }
    
    // Clone method for Strategy interface
    std::unique_ptr<Strategy> clone() const override {
        // Note: This creates a new strategy with the same parameters but empty Q-table
        // For training, you typically don't need to clone the Q-table itself
        auto alphaClone = std::make_unique<EpsilonDecayingParameter>(
            alpha->getValue(), alpha->getValue(), 1.0);
        auto epsilonClone = std::make_unique<EpsilonDecayingParameter>(
            epsilon->getValue(), epsilon->getValue(), 1.0);
        return std::make_unique<QLearningStrategy>(std::move(alphaClone), gamma, std::move(epsilonClone));
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
};
