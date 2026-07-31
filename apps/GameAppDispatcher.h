#pragma once

using GameAppMain = int (*)(int, char**);

int dispatchGameApp(int argc,
                    char** argv,
                    const char* appName,
                    GameAppMain blackjackMain,
                    GameAppMain doubleDownMadnessMain);
