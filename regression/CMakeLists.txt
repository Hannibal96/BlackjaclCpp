# Regression tests as standalone executables (no Google Test)

add_compile_definitions(PROJECT_ROOT="${PROJECT_ROOT_PATH}")

# BasicStrategyRegressionTest
add_executable(BasicStrategyRegressionTest
    ${CMAKE_CURRENT_SOURCE_DIR}/BasicStrategyRegressionTest.cpp
)
target_link_libraries(BasicStrategyRegressionTest
    PRIVATE
        blackjack_objs
)
target_include_directories(BasicStrategyRegressionTest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_property(TARGET BasicStrategyRegressionTest PROPERTY CXX_STANDARD 20)

# QLearningRegressionTest
add_executable(QLearningRegressionTest
    ${CMAKE_CURRENT_SOURCE_DIR}/QLearningRegressionTest.cpp
)
target_link_libraries(QLearningRegressionTest
    PRIVATE
        blackjack_objs
)
target_include_directories(QLearningRegressionTest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_property(TARGET QLearningRegressionTest PROPERTY CXX_STANDARD 20)

add_executable(DoubleDownMadnessEdgeRegressionTest
    ${CMAKE_CURRENT_SOURCE_DIR}/DoubleDownMadnessEdgeRegressionTest.cpp
)
target_link_libraries(DoubleDownMadnessEdgeRegressionTest
    PRIVATE
        blackjack_objs
)
target_include_directories(DoubleDownMadnessEdgeRegressionTest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)
set_property(TARGET DoubleDownMadnessEdgeRegressionTest PROPERTY CXX_STANDARD 20)

add_executable(DoubleDownMadnessQLearningRegressionTest
    ${CMAKE_CURRENT_SOURCE_DIR}/DoubleDownMadnessQLearningRegressionTest.cpp
)
target_link_libraries(DoubleDownMadnessQLearningRegressionTest
    PRIVATE
        blackjack_objs
)
target_include_directories(DoubleDownMadnessQLearningRegressionTest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)
set_property(TARGET DoubleDownMadnessQLearningRegressionTest PROPERTY CXX_STANDARD 20)

add_executable(Spanish21EdgeRegressionTest
    ${CMAKE_CURRENT_SOURCE_DIR}/Spanish21EdgeRegressionTest.cpp
)
target_link_libraries(Spanish21EdgeRegressionTest
    PRIVATE
        blackjack_objs
)
target_include_directories(Spanish21EdgeRegressionTest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)
set_property(TARGET Spanish21EdgeRegressionTest PROPERTY CXX_STANDARD 20)

add_executable(Spanish21QLearningRegressionTest
    ${CMAKE_CURRENT_SOURCE_DIR}/Spanish21QLearningRegressionTest.cpp
)
target_link_libraries(Spanish21QLearningRegressionTest
    PRIVATE
        blackjack_objs
)
target_include_directories(Spanish21QLearningRegressionTest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)
set_property(TARGET Spanish21QLearningRegressionTest PROPERTY CXX_STANDARD 20)
