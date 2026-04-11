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
