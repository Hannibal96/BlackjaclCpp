#pragma once
#include <ostream>

// Action enum representing a player's action
enum class Action {
    HIT,
    STAND,
    DOUBLE_DOWN,
    SPLIT,
    SURRENDER
};

inline std::ostream& operator<<(std::ostream& os, Action action) {
    switch (action) {
        case Action::HIT:          os << "HIT"; break;
        case Action::STAND:        os << "STAND"; break;
        case Action::DOUBLE_DOWN:  os << "DOUBLE_DOWN"; break;
        case Action::SPLIT:        os << "SPLIT"; break;
        case Action::SURRENDER:    os << "SURRENDER"; break;
    }
    return os;
}

