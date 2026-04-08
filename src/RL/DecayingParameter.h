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

    double getValue() const { return value; }

    virtual std::unique_ptr<DecayingParameter> clone() const = 0;
    virtual void updateValue() = 0;

    // Averaging support: accumulate another parameter's mutable state into this one
    virtual DecayingParameter& operator+=(const DecayingParameter& other) = 0;
    // Scale the mutable state (for division after accumulation)
    virtual DecayingParameter& operator*=(double factor) = 0;
};

// Exponential decay parameter (e.g., for epsilon-greedy exploration)
// value *= gamma each step, floored at final_value
class EpsilonDecayingParameter : public DecayingParameter {
private:
    double gamma;

public:
    EpsilonDecayingParameter(double initial_value, double final_value, double gamma)
        : DecayingParameter(initial_value, final_value), gamma(gamma) {}

    void updateValue() override {
        value *= gamma;
        if (value < final_value) value = final_value;
    }

    std::unique_ptr<DecayingParameter> clone() const override {
        return std::make_unique<EpsilonDecayingParameter>(value, final_value, gamma);
    }

    // Only `value` varies per-state; gamma and final_value are fixed hyperparams
    DecayingParameter& operator+=(const DecayingParameter& other) override {
        const auto* o = dynamic_cast<const EpsilonDecayingParameter*>(&other);
        if (o) value += o->value;
        return *this;
    }
    DecayingParameter& operator*=(double factor) override {
        value *= factor;
        return *this;
    }

    double getGamma() const { return gamma; }
    double getFinalValue() const { return final_value; }
};

// Linear decay parameter
// value = N / (n + N), where n counts updates
class LinearDecayingParameter : public DecayingParameter {
private:
    double n;
    double N;

public:
    LinearDecayingParameter(double initial_value, double final_value, int N)
        : DecayingParameter(initial_value, final_value),
          n(N * (1 - initial_value) / initial_value), N(N) {}

    void updateValue() override {
        double v = static_cast<double>(N) / (n + N);
        if (v < final_value) v = final_value;
        this->value = v;
        n++;
    }

    std::unique_ptr<DecayingParameter> clone() const override {
        auto cloned = std::make_unique<LinearDecayingParameter>(value, final_value, N);
        cloned->n = this->n;
        cloned->value = this->value;
        return cloned;
    }

    // Both `value` and `n` vary per-state; N and final_value are fixed hyperparams
    DecayingParameter& operator+=(const DecayingParameter& other) override {
        const auto* o = dynamic_cast<const LinearDecayingParameter*>(&other);
        if (o) { value += o->value; n += o->n; }
        return *this;
    }
    DecayingParameter& operator*=(double factor) override {
        value *= factor;
        n     *= factor;
        return *this;
    }

    int    getN() const { return static_cast<int>(N); }
    double getFinalValue() const { return final_value; }
    double getn() const { return n; }
};
