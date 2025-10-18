# Game Module

This module contains the core game logic and player abstractions for the Blackjack simulator.

## Architecture

### Core Classes

#### `Table` (Abstract Base Class)
Abstract class for managing game flow and table operations.

**Fields:**
- `players` (std::vector<Player*>): Players at the table
- `shoe` (std::unique_ptr<Shoe>): Card shoe (automatically created from rules)

**Constructor:**
- Takes `Rules` and players as parameters
- Extracts `numDecks` and `penetration` from rules to create shoe
- Does NOT store the rules object - only uses it to initialize
- Derived classes extract and store their specific rule fields

**Methods:**
- `round()`: Pure virtual method to execute a complete game round
- `collectBets()`: Protected pure virtual - collect bets from players
- `dealInitialCards()`: Protected pure virtual - deal initial cards
- `playersPlay()`: Protected pure virtual - let players make decisions
- `dealerPlays()`: Protected pure virtual - dealer plays their hand
- `evaluate()`: Protected pure virtual - evaluate hands and process winnings/losses

#### `BlackjackTable` (Concrete Implementation)
Concrete implementation of Table for classic Blackjack.

**Rule Fields (extracted from BlackjackRules):**
- `blackjackPayout`, `standSoft17`, `peekBlackjack`
- `maxSplits`, `doubleAfterSplit`, `resplitAces`, `hitSplitAces`
- `surrender`, `doubleDownOn`

**Implements:**
- Complete round flow (bet → deal → play → dealer → evaluate)
- Player action handling (HIT, STAND, DOUBLE_DOWN)
- Dealer logic based on rule fields
- Hand evaluation and payment calculations

#### `Rules` (Base Struct)
Base rules struct for all game variants.

**Fields:**
- `blackjackPayout` (double): Blackjack payout multiplier (e.g., 1.5 for 3:2, 1.2 for 6:5)
- `standSoft17` (bool): True if dealer stands on soft 17, false if dealer hits
- `numDecks` (int): Number of decks in the shoe
- `penetration` (double): Penetration percentage (0-100)

#### `BlackjackRules` (Struct)
Blackjack-specific rules extending base Rules.

**Additional Fields:**
- `peekBlackjack` (bool): True = dealer peeks for blackjack before players play (American style), False = dealer gets second card after players finish (European style)
- `maxSplits` (unsigned int): Maximum number of splits allowed (e.g., 3 means 4 hands total)
- `doubleAfterSplit` (bool): Can player double down after split
- `resplitAces` (bool): Can player resplit aces
- `hitSplitAces` (bool): Can player hit split aces
- `surrender` (Surrender): Surrender options (NO_SURRENDER, SURRENDER_ANY, SURRENDER_NO_ACE)
- `doubleDownOn` (DoubleDownOn): What totals allow double down (ANY, NINE_TEN_ELEVEN, TEN_ELEVEN)

**Default Rules (American Style):**
- 3:2 blackjack payout
- Dealer stands on soft 17
- 6 decks, 75% penetration
- Dealer peeks for blackjack (American style)
- Maximum 3 splits (4 hands total)
- Can double after split
- Cannot resplit aces
- Cannot hit split aces
- Surrender allowed on any hand
- Can double on any total

**Enums:**
- `DoubleDownOn::ANY` - Can double on any total
- `DoubleDownOn::NINE_TEN_ELEVEN` - Can only double on 9, 10, or 11
- `DoubleDownOn::TEN_ELEVEN` - Can only double on 10 or 11
- `Surrender::NO_SURRENDER` - Cannot surrender
- `Surrender::SURRENDER_ANY` - Can surrender on any hand
- `Surrender::SURRENDER_NO_ACE` - Can surrender only if no ace in hand

#### `Player` (Abstract Base Class)
Abstract class representing a blackjack player.

**Fields:**
- `money` (double): Player's current money/bankroll
- `strategy` (std::unique_ptr<Strategy>): Strategy for decision-making

**Methods:**
- `getAction(const State& state)`: Pure virtual method to get player's action based on game state
- `getBet()`: Pure virtual method to get player's bet amount
- `updateMoney(double amount)`: Update player's money (add/subtract)
- `getMoney()`: Get current money
- `setStrategy(std::unique_ptr<Strategy>)`: Change the player's strategy

#### `Strategy` (Abstract Base Class)
Abstract class defining the interface for decision-making strategies.

**Note:** Every Player must have a Strategy. The constructor will throw an exception if no strategy is provided.

**Methods:**
- `getAction(const State& state)`: Get action based on state

**Note:** Betting logic is separate and will be handled by a different class.

### Data Structures

#### `Hand`
Represents a player's hand in blackjack.

**Fields:**
- `cards`: Vector of cards in the hand
- `bet`: Bet amount for this hand
- `isSplit`: Flag indicating if this hand is from a split

**Methods:**
- `addCard(const Card&)`: Add a card to the hand
- `getCards()`: Get all cards
- `getBet()` / `setBet()`: Get/set bet amount
- `getIsSplit()` / `setIsSplit()`: Get/set split flag
- `getValue()`: Calculate hand value (handles soft/hard totals)
- `isPair()`: Check if hand is a pair (for split logic)
- `isBlackjack()`: Check if hand is natural blackjack
- `isBust()`: Check if hand is bust
- `clear()`: Reset the hand

#### `State`
Represents the current game state passed to the player for decision-making.

**Fields:**
- `playerHand`: Hand object containing player's cards, bet, and split status (`getIsSplit()`)
- `dealerCard`: The dealer's visible card

**Note:** Game logic like canSplit, canDoubleDown, etc. are determined by table rules, not stored in State. The `playerHand` includes an `isSplit` flag accessible via `getIsSplit()`.

#### `Action`
Enum representing a player's action.

**Values:**
- `HIT`: Request another card
- `STAND`: Keep current hand
- `DOUBLE_DOWN`: Double bet and take one more card
- `SPLIT`: Split pair into two hands
- `SURRENDER`: Forfeit half the bet and end the hand

## Usage Example

### Complete Game Setup

```cpp
#include "Game/BlackjackTable.h"
#include "Game/BlackjackRules.h"
#include "Game/Player.h"
#include "Game/Strategy.h"

// Create rules (uses default classic blackjack rules)
BlackjackRules rules;

// Or customize rules (e.g., European style)
// BlackjackRules rules(1.2, false, 8, 80.0, false, 3, true, false, false, 
//                      Surrender::SURRENDER_NO_ACE, DoubleDownOn::TEN_ELEVEN);
// 6:5 payout, dealer hits soft 17, 8 decks, 80% penetration, 
// no peek (European), can surrender without ace, can double on 10 or 11 only

// Create strategy
class MyStrategy : public Strategy {
public:
    Action getAction(const State& state) override {
        int handValue = state.playerHand.getValue();
        if (handValue < 17) {
            return Action::HIT;
        }
        return Action::STAND;
    }
};

// Create player
class MyPlayer : public Player {
public:
    MyPlayer(double initialMoney, std::unique_ptr<Strategy> strat)
        : Player(initialMoney, std::move(strat)) {}
    
    Action getAction(const State& state) override {
        return strategy->getAction(state);
    }
    
    unsigned int getBet() override {
        return 10; // Flat betting
    }
};

auto strategy = std::make_unique<MyStrategy>();
MyPlayer player(1000.0, std::move(strategy));

// Create table (shoe is automatically created based on rules)
std::vector<Player*> players = {&player};
BlackjackTable table(rules, players);

// Play rounds
for (int i = 0; i < 100; ++i) {
    table.round();
}

std::cout << "Final bankroll: " << player.getMoney() << std::endl;
```

### Creating Custom Strategy

```cpp
#include "Game/Player.h"
#include "Game/Strategy.h"

// Create a custom strategy
class MyStrategy : public Strategy {
public:
    Action getAction(const State& state) override {
        int handValue = state.playerHand.getValue();
        if (handValue < 17) {
            return Action::HIT;
        }
        return Action::STAND;
    }
};

// Create a custom player by inheriting from abstract Player class
class MyPlayer : public Player {
public:
    MyPlayer(double initialMoney, std::unique_ptr<Strategy> strat)
        : Player(initialMoney, std::move(strat)) {}
    
    Action getAction(const State& state) override {
        return strategy->getAction(state);
    }
    
    unsigned int getBet() override {
        // Betting logic implemented here (not part of Strategy)
        return 10; // Example: flat betting
    }
};

// Use the player in game
auto strategy = std::make_unique<MyStrategy>();
MyPlayer player(1000.0, std::move(strategy));

State gameState;
// ... populate gameState ...

Action action = player.getAction(gameState);
unsigned int bet = player.getBet();

// Update money after round
player.updateMoney(-50.0); // Lost 50
// or
player.updateMoney(100.0); // Won 100
```

## Extending the System

### Creating Custom Players
1. Inherit from `Player`
2. Implement `getAction()` and `getBet()`
3. Add any custom logic or fields

### Creating Custom Strategies
1. Inherit from `Strategy`
2. Implement `getAction()`
3. Can be used with any Player implementation

**Note:** Betting logic is implemented in the Player class, not in Strategy.

### Creating Custom Game Variants
1. **Create Custom Rules:**
   - Create a struct extending `Rules`
   - Add variant-specific fields
   - Example: `DoubleDownMadnessRules`

2. **Create Custom Table:**
   - Inherit from `Table`
   - Implement `round()` and protected methods
   - Add variant-specific logic
   - Example: `DoubleDownMadnessTable`

```cpp
// Example: Custom rules variant
struct SpanishBlackjackRules : public Rules {
    bool remove10s;
    // ... other variant-specific fields
    
    SpanishBlackjackRules()
        : Rules(1.5, true, 6, 75.0),
          remove10s(true)
    {}
};

// Example: Custom table variant
class SpanishBlackjackTable : public Table {
private:
    // Extract and store rule fields needed
    bool remove10s;
    double blackjackPayout;
    
public:
    SpanishBlackjackTable(const SpanishBlackjackRules& rules, 
                          std::vector<Player*> players)
        : Table(rules, players),  // Base creates shoe from rules
          remove10s(rules.remove10s),
          blackjackPayout(rules.blackjackPayout) {}
    
    void round() override {
        // Custom game flow using stored rule fields
    }
    // ... implement other methods
};
```

This architecture allows for:
- Strategy pattern for flexible decision-making
- Easy testing of different strategies
- Integration with RL agents (implement Strategy interface)
- Separation of player state and decision logic
- Multiple game variants with shared infrastructure
- Extensible rules system

