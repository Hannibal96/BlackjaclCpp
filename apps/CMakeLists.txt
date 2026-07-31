add_compile_definitions(PROJECT_ROOT="${PROJECT_ROOT_PATH}")

# FindDeviations — Q-learning trainer with Hi-Lo card counting
add_executable(FindDeviations
    ${CMAKE_CURRENT_SOURCE_DIR}/FindDeviations.cpp
)
target_link_libraries(FindDeviations
    PRIVATE
        $<TARGET_OBJECTS:blackjack_objs>
)
target_include_directories(FindDeviations
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/unittest/regression   # RegressionTestUtils.h
)
set_property(TARGET FindDeviations PROPERTY CXX_STANDARD 20)

# MeasureEdge — measure the edge of a card counter under a fixed BasicStrategy
add_executable(MeasureEdge
    ${CMAKE_CURRENT_SOURCE_DIR}/MeasureEdge.cpp
)
target_link_libraries(MeasureEdge
    PRIVATE
        $<TARGET_OBJECTS:blackjack_objs>
)
target_include_directories(MeasureEdge
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/unittest/regression   # RegressionTestUtils.h
)
set_property(TARGET MeasureEdge PROPERTY CXX_STANDARD 20)

# FindOptimalCount — OLS regression to derive optimal card counting weights
add_executable(FindOptimalCount
    ${CMAKE_CURRENT_SOURCE_DIR}/FindOptimalCount.cpp
)
target_link_libraries(FindOptimalCount
    PRIVATE
        $<TARGET_OBJECTS:blackjack_objs>
)
target_include_directories(FindOptimalCount
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/unittest/regression   # RegressionTestUtils.h
)
set_property(TARGET FindOptimalCount PROPERTY CXX_STANDARD 20)

# AlternatingOptimization — alternate between RL policy learning and OLS count learning
add_executable(AlternatingOptimization
    ${CMAKE_CURRENT_SOURCE_DIR}/AlternatingOptimization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GameAppDispatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BlackjackAlternatingOptimization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DoubleDownMadnessAlternatingOptimization.cpp
)
target_link_libraries(AlternatingOptimization
    PRIVATE
        $<TARGET_OBJECTS:blackjack_objs>
)
target_include_directories(AlternatingOptimization
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/unittest/regression
)
set_property(TARGET AlternatingOptimization PROPERTY CXX_STANDARD 20)

# CompareCountStrategies — compare basic strategy, Illustrious 18, and full deviations for a count
add_executable(CompareCountStrategies
    ${CMAKE_CURRENT_SOURCE_DIR}/CompareCountStrategies.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GameAppDispatcher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BlackjackCompareCountStrategies.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DoubleDownMadnessCompareCountStrategies.cpp
)
target_link_libraries(CompareCountStrategies
    PRIVATE
        $<TARGET_OBJECTS:blackjack_objs>
)
target_include_directories(CompareCountStrategies
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/unittest/regression
)
set_property(TARGET CompareCountStrategies PROPERTY CXX_STANDARD 20)
