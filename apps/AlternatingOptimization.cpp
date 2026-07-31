#include "GameAppDispatcher.h"

int runBlackjackAlternatingOptimization(int argc, char** argv);
int runDoubleDownMadnessAlternatingOptimization(int argc, char** argv);

int main(int argc, char** argv) {
    return dispatchGameApp(
        argc,
        argv,
        "AlternatingOptimization",
        runBlackjackAlternatingOptimization,
        runDoubleDownMadnessAlternatingOptimization);
}
