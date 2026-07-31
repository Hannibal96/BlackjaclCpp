#pragma once

#include "BasicStrategy.h"
#include "Game/DoubleDownMadnessRules.h"
#include <memory>

std::unique_ptr<BasicStrategy> loadDoubleDownMadnessBasicStrategy(
    DoubleDownMadnessVersion version);
