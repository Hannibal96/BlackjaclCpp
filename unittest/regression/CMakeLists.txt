# Regression tests as standalone executables (no Google Test)

add_compile_definitions(PROJECT_ROOT="${PROJECT_ROOT_PATH}")

# BasicStrategyRegressionTest
add_executable(BasicStrategyRegressionTest
    ${CMAKE_CURRENT_SOURCE_DIR}/BasicStrategyRegressionTest.cpp
)
target_link_libraries(BasicStrategyRegressionTest
    PRIVATE
        $<TARGET_OBJECTS:blackjack_objs>
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
        $<TARGET_OBJECTS:blackjack_objs>
)
target_include_directories(QLearningRegressionTest
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_property(TARGET QLearningRegressionTest PROPERTY CXX_STANDARD 20)

