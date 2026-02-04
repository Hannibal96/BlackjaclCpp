#pragma once

// Abstract base class for decaying parameters used in reinforcement learning
class DecayingParameter {
protected:
    double value;
    double final_value;
    
public:
    DecayingParameter(double initial_value, double final_value) 
        : value(initial_value), final_value(final_value) {}
    
    virtual ~DecayingParameter() = default;
    
    // Get the current value of the parameter
    double getValue() const {
        return value;
    }
    
    // Clone the parameter with current state
    virtual std::unique_ptr<DecayingParameter> clone() const = 0;
    
    // Update the value according to the specific decay strategy
    virtual void updateValue() = 0;
};

// Exponential decay parameter (e.g., for epsilon-greedy exploration)
// Updates as: value *= gamma
class EpsilonDecayingParameter : public DecayingParameter {
private:
    double gamma;  // Decay rate (should be < 1.0 for decay)
    
public:
    EpsilonDecayingParameter(double initial_value, double final_value, double gamma)
        : DecayingParameter(initial_value, final_value), gamma(gamma) {}
    
    void updateValue() override {
        value *= gamma;
        // Ensure value doesn't go below final_value
        if (value < final_value) {
            value = final_value;
        }
    }
    
    std::unique_ptr<DecayingParameter> clone() const override {
        return std::make_unique<EpsilonDecayingParameter>(value, final_value, gamma);
    }
    
    double getGamma() const { return gamma; }
    double getFinalValue() const { return final_value; }
};

// Linear decay parameter
// Updates as: value = n / (n + N) where n is the number of updates
class LinearDecayingParameter : public DecayingParameter {
private:
    double n;     // Number of times update has been called
    double N;     // Parameter controlling decay rate
    
public:
    LinearDecayingParameter(double initial_value, double final_value, int N)
        : DecayingParameter(initial_value, final_value), 
          n(N * (1-initial_value) / initial_value), N(N){          }
    
    void updateValue() override {
        // Calculate new value using linear decay formula
        double value = static_cast<double>(N) / (n + N);
        
        // Ensure value doesn't go below final_value
        if (value < final_value) {
            value = final_value;
        }
        this->value = value;
        n++;
    }
    
    std::unique_ptr<DecayingParameter> clone() const override {
        auto cloned = std::make_unique<LinearDecayingParameter>(value, final_value, N);
        cloned->n = this->n;  // Preserve decay state
        cloned->value = this->value;
        return cloned;
    }
    
    int getN() const { return N; }
    double getFinalValue() const { return final_value; }
};
