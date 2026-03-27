cmake_minimum_required(VERSION 3.20)

project(BlackjaclCpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# put test executables in build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" PROJECT_ROOT_PATH)
add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# compile production code once (no main) as an object library
add_library(blackjack_objs OBJECT
    src/Shoe/Deck.cpp
    src/Shoe/Shoe.cpp
    src/Game/Hand.cpp
    src/Game/Slot.cpp
    src/Game/Player.cpp
    src/Game/Table.cpp
    src/Game/BlackjackTable.cpp
    src/RL/RandomStrategy.cpp
    src/RL/BasicStrategy.cpp
    src/Utils/Utils.cpp
    src/Utils/GlobalRNG.cpp
)
target_include_directories(blackjack_objs
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# testing from the top level
include(CTest)
enable_testing()

# NOTE: your tests dir is named 'unittest'
add_subdirectory(unittest)

