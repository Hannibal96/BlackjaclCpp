# GoogleTest via FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# helper: define a gtest target and link to your production objects
function(add_gtest TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    target_link_libraries(${TARGET_NAME}
        PRIVATE
            GTest::gtest_main
            $<TARGET_OBJECTS:blackjack_objs>
    )
    target_include_directories(${TARGET_NAME}
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/include
    )
    set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 20)
    gtest_discover_tests(${TARGET_NAME})
endfunction()

add_compile_definitions(PROJECT_ROOT="${PROJECT_ROOT_PATH}")

# list ONLY the test .cpp files here; production .cpp are in blackjack_objs
add_gtest(ShoeTest
    ${CMAKE_CURRENT_SOURCE_DIR}/ShoeTest.cpp
)

add_gtest(HandTest
    ${CMAKE_CURRENT_SOURCE_DIR}/HandTest.cpp
)

add_gtest(BlackjackTableTest
    ${CMAKE_CURRENT_SOURCE_DIR}/BlackjackTableTest.cpp
)

add_gtest(BasicStrategyRegressionTest
    ${CMAKE_CURRENT_SOURCE_DIR}/BasicStrategyRegressionTest.cpp
)
target_include_directories(BasicStrategyRegressionTest PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

add_gtest(PerformanceBenchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/PerformanceBenchmark.cpp
)

add_gtest(BlackjackBenchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/BlackjackBenchmark.cpp
)

add_gtest(DebugShoeTest
    ${CMAKE_CURRENT_SOURCE_DIR}/DebugShoeTest.cpp
)

# convenient meta-target to build all tests
add_custom_target(tests ALL
  DEPENDS
    ShoeTest
    HandTest
    BlackjackTableTest
    BasicStrategyRegressionTest
    PerformanceBenchmark
    BlackjackBenchmark
    DebugShoeTest
)
