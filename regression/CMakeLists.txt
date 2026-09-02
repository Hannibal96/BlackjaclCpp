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

add_executable(CountQuantizationTest
    ${CMAKE_CURRENT_SOURCE_DIR}/CountQuantizationTest.cpp
)
target_link_libraries(CountQuantizationTest
    PRIVATE
        blackjack_objs
)
target_include_directories(CountQuantizationTest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)
set_property(TARGET CountQuantizationTest PROPERTY CXX_STANDARD 20)

add_executable(KellyExposureTest
    ${CMAKE_CURRENT_SOURCE_DIR}/KellyExposureTest.cpp
)
target_link_libraries(KellyExposureTest
    PRIVATE
        blackjack_objs
)
target_include_directories(KellyExposureTest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)
set_property(TARGET KellyExposureTest PROPERTY CXX_STANDARD 20)

add_executable(CountPolicyEvaluationTest
    ${CMAKE_CURRENT_SOURCE_DIR}/CountPolicyEvaluationTest.cpp
)
target_link_libraries(CountPolicyEvaluationTest
    PRIVATE
        blackjack_objs
)
target_include_directories(CountPolicyEvaluationTest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_property(TARGET CountPolicyEvaluationTest PROPERTY CXX_STANDARD 20)
