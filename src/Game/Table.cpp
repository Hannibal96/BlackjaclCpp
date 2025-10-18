#include "Table.h"
#include "../Shoe/Shoe.h"
#include <stdexcept>

// Constructor - receives rules, creates shoe, does NOT store rules
Table::Table(const Rules& gameRules, std::vector<Player*> gamePlayers)
    : players(std::move(gamePlayers))
{
    if (players.empty()) {
        throw std::invalid_argument("Table must have at least one player");
    }
    
    // Create shoe based on rules (extract what we need)
    shoe = std::make_unique<Shoe>(gameRules.numDecks, gameRules.penetration);
}

