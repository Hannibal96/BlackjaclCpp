add_compile_definitions(PROJECT_ROOT="${PROJECT_ROOT_PATH}")

# AlternatingOptimization — alternate between RL policy learning and count
# regression, for both classic blackjack and Double Down Madness (--game).
# Game-specific behavior lives in GameTraits.h; AlternatingOptimization.cpp
# holds the shared AlternatingOptimizationApp<Game> engine, its two
# per-game instantiations, and main().
add_executable(AlternatingOptimization
    ${CMAKE_CURRENT_SOURCE_DIR}/AlternatingOptimization.cpp
)
target_link_libraries(AlternatingOptimization
    PRIVATE
        $<TARGET_OBJECTS:blackjack_objs>
)
target_include_directories(AlternatingOptimization
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/regression
)
set_property(TARGET AlternatingOptimization PROPERTY CXX_STANDARD 20)

# CompareCountStrategies — compare basic strategy (plus Illustrious 18 for
# blackjack) against full deviations trained fresh or loaded from an
# AlternatingOptimization checkpoint, for both games (--game). Same shape as
# AlternatingOptimization.cpp: engine + instantiations + main() in one file.
add_executable(CompareCountStrategies
    ${CMAKE_CURRENT_SOURCE_DIR}/CompareCountStrategies.cpp
)
target_link_libraries(CompareCountStrategies
    PRIVATE
        $<TARGET_OBJECTS:blackjack_objs>
)
target_include_directories(CompareCountStrategies
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/regression
)
set_property(TARGET CompareCountStrategies PROPERTY CXX_STANDARD 20)
