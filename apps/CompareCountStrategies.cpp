#include "GameAppDispatcher.h"

int runBlackjackCompareCountStrategies(int argc, char** argv);
int runDoubleDownMadnessCompareCountStrategies(int argc, char** argv);

int main(int argc, char** argv) {
    return dispatchGameApp(
        argc,
        argv,
        "CompareCountStrategies",
        runBlackjackCompareCountStrategies,
        runDoubleDownMadnessCompareCountStrategies);
}
