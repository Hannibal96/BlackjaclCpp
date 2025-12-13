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
};

// Linear decay parameter
// Updates as: value = n / (n + N) where n is the number of updates
class LinearDecayingParameter : public DecayingParameter {
private:
    int n;     // Number of times update has been called
    int N;     // Parameter controlling decay rate
    double initial_value;
    
public:
    LinearDecayingParameter(double initial_value, double final_value, int N)
        : DecayingParameter(initial_value, final_value), 
          n(0), N(N), initial_value(initial_value) {}
    
    void updateValue() override {
        n++;
        // Calculate new value using linear decay formula
        double value = static_cast<double>(n) / (n + N);
        
        // Ensure value doesn't go below final_value
        if (value < final_value) {
            value = final_value;
        }
    }
};
