#include "Deck.h"

// Card methods
int Card::getValue() const {
    switch (rank) {
        case Rank::TWO:   return 2;
        case Rank::THREE: return 3;
        case Rank::FOUR:  return 4;
        case Rank::FIVE:  return 5;
        case Rank::SIX:   return 6;
        case Rank::SEVEN: return 7;
        case Rank::EIGHT: return 8;
        case Rank::NINE:  return 9;
        case Rank::TEN:
        case Rank::JACK:
        case Rank::QUEEN:
        case Rank::KING:  return 10;
        case Rank::ACE:   return 11;  // Ace is 11 by default, game logic handles soft/hard hands
        default:          return 0;
    }
}

std::string Card::toString() const {
    std::string rankStr;
    switch (rank) {
        case Rank::TWO:   rankStr = "2"; break;
        case Rank::THREE: rankStr = "3"; break;
        case Rank::FOUR:  rankStr = "4"; break;
        case Rank::FIVE:  rankStr = "5"; break;
        case Rank::SIX:   rankStr = "6"; break;
        case Rank::SEVEN: rankStr = "7"; break;
        case Rank::EIGHT: rankStr = "8"; break;
        case Rank::NINE:  rankStr = "9"; break;
        case Rank::TEN:   rankStr = "10"; break;
        case Rank::JACK:  rankStr = "J"; break;
        case Rank::QUEEN: rankStr = "Q"; break;
        case Rank::KING:  rankStr = "K"; break;
        case Rank::ACE:   rankStr = "A"; break;
    }
    
    std::string suitStr;
    switch (suit) {
        case Suit::HEARTS:   suitStr = "♥"; break;
        case Suit::DIAMONDS: suitStr = "♦"; break;
        case Suit::CLUBS:    suitStr = "♣"; break;
        case Suit::SPADES:   suitStr = "♠"; break;
    }
    
    return rankStr + suitStr;
}

// Deck methods
Deck::Deck(bool spanishDeck) : spanish(spanishDeck) {
    initialize();
}

void Deck::initialize() {
    cards.clear();

    // Create all cards (13 ranks x 4 suits, or 12 ranks x 4 suits for a Spanish deck
    // that omits rank TEN — J/Q/K remain and still count as 10).
    for (int suit = 0; suit < 4; ++suit) {
        for (int rank = 2; rank <= 14; ++rank) {  // 2-10, 11=J, 12=Q, 13=K, 14=A
            if (spanish && rank == 10) continue;  // Spanish deck has no rank-TEN cards

            Rank cardRank;
            switch (rank) {
                case 2:  cardRank = Rank::TWO; break;
                case 3:  cardRank = Rank::THREE; break;
                case 4:  cardRank = Rank::FOUR; break;
                case 5:  cardRank = Rank::FIVE; break;
                case 6:  cardRank = Rank::SIX; break;
                case 7:  cardRank = Rank::SEVEN; break;
                case 8:  cardRank = Rank::EIGHT; break;
                case 9:  cardRank = Rank::NINE; break;
                case 10: cardRank = Rank::TEN; break;
                case 11: cardRank = Rank::JACK; break;
                case 12: cardRank = Rank::QUEEN; break;
                case 13: cardRank = Rank::KING; break;
                case 14: cardRank = Rank::ACE; break;
                default: cardRank = Rank::TWO; break;
            }

            Suit cardSuit = static_cast<Suit>(suit);
            cards.emplace_back(cardRank, cardSuit);
        }
    }
}

