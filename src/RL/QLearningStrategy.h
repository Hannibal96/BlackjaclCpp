#pragma once
#include "Strategy.h"
#include "DecayingParameter.h"
#include "BasicStrategy.h"
#include <map>
#include <memory>
#include <random>
#include <algorithm>
#include <cmath>
#include <string>
#include <nlohmann/json.hpp>

enum class ExplorationMode { EPSILON_GREEDY, BOLTZMANN };

// Q-Learning strategy for reinforcement learning
class QLearningStrategy : public Strategy {
private:
    using QTableKey = std::pair<StateKey, Action>;
    std::map<QTableKey, double> qTable;

    // Per-state learning parameters
    std::map<StateKey, std::unique_ptr<DecayingParameter>> alphaMap;
    std::map<StateKey, std::unique_ptr<DecayingParameter>> explorationMap;

    // Template parameters for creating new state entries
    std::unique_ptr<DecayingParameter> alphaTemplate;
    std::unique_ptr<DecayingParameter> explorationTemplate;

    double gamma;
    ExplorationMode explorationMode;

    mutable std::mt19937 rng;
    mutable std::uniform_real_distribution<double> dist;

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

    // Get Q-value for a key-action pair (returns 0 if not in table)
    double getQValue(const StateKey& key, Action action) const {
        auto it = qTable.find(std::make_pair(key, action));
        return (it != qTable.end()) ? it->second : 0.0;
    }

    // Get the maximum Q-value for a given key across all allowed actions
    double getMaxQValue(const StateKey& key, const std::vector<Action>& allowedActions) const {
        double maxQ = -std::numeric_limits<double>::infinity();
        bool foundAction = false;

        for (const Action& action : allowedActions) {
            double q = getQValue(key, action);
            if (!foundAction || q > maxQ) {
                maxQ = q;
                foundAction = true;
            }
        }

        return foundAction ? maxQ : 0.0;
    }

    // Get the greedy action (action with highest Q-value)
    Action getGreedyAction(const StateKey& key, const std::vector<Action>& allowedActions) const {
        Action bestAction = allowedActions[0];
        double maxQ = getQValue(key, bestAction);

        for (const Action& action : allowedActions) {
            double q = getQValue(key, action);
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

    Action getAction(const StateKey& key, const std::vector<Action>& allowedActions) override {
        DecayingParameter& stateExploration = getOrCreateParam(
            key, explorationMap, explorationTemplate);

        if (explorationMode == ExplorationMode::EPSILON_GREEDY) {
            if (dist(rng) < stateExploration.getValue()) {
                stateExploration.updateValue();
                std::uniform_int_distribution<size_t> actionDist(0, allowedActions.size() - 1);
                return allowedActions[actionDist(rng)];
            } else {
                return getGreedyAction(key, allowedActions);
            }
        } else {
            // Boltzmann (softmax): P(a|s) = exp(Q(s,a)/τ) / Σ exp(Q(s,b)/τ)
            double temperature = stateExploration.getValue();

            std::vector<double> qValues;
            qValues.reserve(allowedActions.size());
            for (const Action& action : allowedActions) {
                qValues.push_back(getQValue(key, action));
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
            for (size_t i = 0; i < allowedActions.size(); ++i) {
                cumulative += weights[i];
                if (r <= cumulative) {
                    stateExploration.updateValue();
                    return allowedActions[i];
                }
            }
            stateExploration.updateValue();
            return allowedActions.back();
        }
    }

    // Update Q-table using Q-learning update rule:
    // Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
    void updateTable(const StateKey& currentKey, Action action, double reward,
                     const StateKey& nextKey, const std::vector<Action>& nextAllowedActions) override {
        DecayingParameter& stateAlpha = getOrCreateParam(currentKey, alphaMap, alphaTemplate);

        double currentQ = getQValue(currentKey, action);
        double maxNextQ = getMaxQValue(nextKey, nextAllowedActions);

        // Special handling for SPLIT: we play two hands, so expected value doubles
        if (action == Action::SPLIT) {
            maxNextQ *= 2.0;
        }

        double newQ = currentQ + stateAlpha.getValue() * (reward + gamma * maxNextQ - currentQ);
        qTable[std::make_pair(currentKey, action)] = newQ;
        stateAlpha.updateValue();
    }

    void updateAlpha(const StateKey& key) {
        getOrCreateParam(key, alphaMap, alphaTemplate).updateValue();
    }

    void updateExploration(const StateKey& key) {
        getOrCreateParam(key, explorationMap, explorationTemplate).updateValue();
    }

    double getAlpha(const StateKey& key) {
        return getOrCreateParam(key, alphaMap, alphaTemplate).getValue();
    }

    double getExploration(const StateKey& key) {
        return getOrCreateParam(key, explorationMap, explorationTemplate).getValue();
    }

    double getTemplateAlpha() const {
        return alphaTemplate->getValue();
    }

    double getTemplateExploration() const {
        return explorationTemplate->getValue();
    }

    ExplorationMode getExplorationMode() const { return explorationMode; }

    size_t getTableSize() const {
        return qTable.size();
    }

    void printTo(std::ostream& os) const override {
        auto basicStrategy = toBasicStrategy();
        os << *basicStrategy;
    }

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

        std::map<StateKey, std::vector<std::pair<Action, double>>> stateActions;
        for (const auto& [key, qValue] : qTable) {
            const auto& [stateKey, action] = key;
            stateActions[stateKey].push_back({action, qValue});
        }

        for (auto& [stateKey, actions] : stateActions) {
            std::sort(actions.begin(), actions.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });

            if (actions.empty()) continue;

            Action bestAction = actions[0].first;
            Action fallbackAction = bestAction;

            if (bestAction == Action::SURRENDER || bestAction == Action::DOUBLE_DOWN || bestAction == Action::SPLIT) {
                for (size_t i = 1; i < actions.size(); ++i) {
                    Action candidate = actions[i].first;
                    if (candidate == Action::HIT || candidate == Action::STAND) {
                        fallbackAction = candidate;
                        break;
                    }
                }
                if (fallbackAction == bestAction) {
                    fallbackAction = Action::HIT;
                }
            }

            const auto& [count, handType, playerSum, dealerHand] = stateKey;
            basicStrategy->setAction(count, handType, playerSum, dealerHand,
                ActionWithFallback(bestAction, fallbackAction));
        }

        return basicStrategy;
    }

    Strategy& operator+=(const Strategy& other) override {
        const QLearningStrategy* otherQL = dynamic_cast<const QLearningStrategy*>(&other);
        if (!otherQL) return *this;

        for (const auto& [key, value] : otherQL->qTable)
            qTable[key] += value;

        for (const auto& [key, param] : otherQL->alphaMap) {
            auto it = alphaMap.find(key);
            if (it != alphaMap.end()) *it->second += *param;
            else                       alphaMap[key] = param->clone();
        }

        for (const auto& [key, param] : otherQL->explorationMap) {
            auto it = explorationMap.find(key);
            if (it != explorationMap.end()) *it->second += *param;
            else                             explorationMap[key] = param->clone();
        }

        return *this;
    }

    Strategy& operator*=(double factor) override {
        for (auto& [key, value] : qTable)
            value *= factor;
        for (auto& [key, param] : alphaMap)
            *param *= factor;
        for (auto& [key, param] : explorationMap)
            *param *= factor;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const QLearningStrategy& strategy) {
        auto basicStrategy = strategy.toBasicStrategy();
        os << *basicStrategy;
        return os;
    }

    // ----- Checkpoint serialization -----

    // Serialize full agent state (Q-table + per-state maps + templates) to JSON
    nlohmann::json toJson(uint64_t roundsCompleted = 0) const;

    // Save agent state to file; also writes a .meta.json sidecar if metaJson is non-null
    bool saveToFile(const std::string& filepath,
                    uint64_t roundsCompleted = 0,
                    const nlohmann::json* metaJson = nullptr) const;

    // Deserialize agent state from JSON produced by toJson()
    static std::unique_ptr<QLearningStrategy> fromJson(const nlohmann::json& j);

    // Load agent state from a checkpoint file
    static std::unique_ptr<QLearningStrategy> loadFromFile(const std::string& filepath);
};
