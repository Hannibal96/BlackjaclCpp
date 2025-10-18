# Unit Tests for Blackjack Simulator

This directory contains unit tests for the Blackjack Simulator components.

## Framework

The tests use **Google Test (gtest)** framework, which is automatically downloaded via CMake's FetchContent.

## Test Coverage

### HandTest.cpp

Comprehensive tests for the `Hand` class:

1. **Constructor Tests**
   - Default constructor
   - Constructor with bet amount

2. **Card Management Tests**
   - Add cards to hand
   - Get cards from hand
   - Card count

3. **Hand Value Tests - Hard Hands**
   - Simple hard hands (no Aces)
   - Multiple cards
   - Various combinations

4. **Hand Value Tests - Soft Hands**
   - Ace as 11
   - Ace with ten-value cards
   - Ace converting from 11 to 1 when busting

5. **Multiple Aces Tests**
   - Two Aces (one as 11, one as 1)
   - Three Aces
   - Four Aces
   - Multiple Aces with other cards

6. **isPair() Tests**
   - Valid pairs (same rank)
   - Ten-value pairs (K+Q, K+J, etc.)
   - Not pairs (different values)
   - Not pairs (more than 2 cards)

7. **isBlackjack() Tests**
   - Blackjack with various combinations (A+K, A+Q, A+10)
   - 21 with 3 cards (not blackjack)
   - Non-21 hands

8. **isBust() Tests**
   - Bust over 21
   - Not bust at 21
   - Not bust under 21
   - Not bust with Ace conversion

9. **Bet Management Tests**
   - Set and get bet
   - Multiply bet (default 2x)
   - Multiply bet with custom multiplier
   - Edge cases

10. **Split Flag Tests**
    - Get and set isSplit flag

11. **Clear Method Tests**
    - Clear resets all fields

12. **Edge Cases**
    - Empty hand value
    - Single card values
    - Maximum value scenarios

### ShoeTest.cpp

Comprehensive tests for the `Shoe` class:

1. **Constructor Tests**
   - Valid parameters (various deck numbers and penetration levels)
   - Invalid number of decks (zero, negative)
   - Invalid penetration values (negative, >100%)

2. **Card Count Tests**
   - Total cards equals num_decks × 52
   - Cards remaining decreases after dealing
   - Single deck has 52 unique cards
   - Multiple decks have correct card distribution

3. **Dealing Tests**
   - Deal card returns valid cards
   - Card values are correct (2-11)
   - Cannot deal more cards than available (throws exception)

4. **Penetration Tests**
   - Penetration threshold triggers end_shoe flag
   - Different penetration levels (50%, 75%, 100%)
   - End_shoe flag is initially false

5. **Shuffle and Reset Tests**
   - Shuffle randomizes card order
   - Reset restores all cards
   - Reset clears end_shoe flag

## Building and Running Tests

### Using CMake (Command Line)

```bash
# Configure and build
cmake -B build
cmake --build build

# Run all tests
ctest --test-dir build

# Or run the test executable directly
./build/unittest/ShoeTest
```

### Using Visual Studio

1. Open the project in Visual Studio
2. Build the solution (this will build the tests)
3. Open **Test Explorer** (Test → Test Explorer)
4. Click "Run All" to execute all tests

### Using Cursor

1. CMake should automatically configure the project
2. The test executable will be built: `out/build/x64-debug/unittest/ShoeTest.exe`
3. Run via terminal or use the test integration if available

## Adding New Tests

To add new test files:

1. Create a new `.cpp` file in the `unittest` directory
2. Include necessary headers and Google Test
3. Add the test file to `unittest/CMakeLists.txt`:
   ```cmake
   add_executable(YourTest
       YourTest.cpp
       # ... source files needed
   )
   target_link_libraries(YourTest GTest::gtest_main)
   gtest_discover_tests(YourTest)
   ```

## Test Results

All tests should pass with properly implemented classes. Example output:

```
[==========] Running 50+ tests from 2 test suites.
[----------] Global test environment set-up.
[----------] 35 tests from HandTest
[ RUN      ] HandTest.DefaultConstructor
[       OK ] HandTest.DefaultConstructor
[ RUN      ] HandTest.SoftHandAceAs11
[       OK ] HandTest.SoftHandAceAs11
...
[----------] 35 tests from HandTest (XX ms total)

[----------] 15 tests from ShoeTest
[ RUN      ] ShoeTest.ConstructorValidParameters
[       OK ] ShoeTest.ConstructorValidParameters
...
[----------] 15 tests from ShoeTest (XX ms total)

[==========] 50 tests from 2 test suites ran. (XX ms total)
[  PASSED  ] 50 tests.
```

