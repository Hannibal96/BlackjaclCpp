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
