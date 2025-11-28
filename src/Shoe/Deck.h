#pragma once
#include <ostream>
#include <vector>
#include <string>

enum class Rank { TWO = 2, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE };
enum class Suit { DIAMONDS, HEARTS, CLUBS, SPADES };

// Mark inline if this lives in a header to avoid ODR issues
inline std::ostream& operator<<(std::ostream& os, Suit s) {
    switch (s) {
        case Suit::HEARTS:   os << "h"; break;
        case Suit::DIAMONDS: os << "d"; break;
        case Suit::CLUBS:    os << "c"; break;
        case Suit::SPADES:   os << "s"; break;
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, Rank r) {
    switch (r) {
        case Rank::TWO:   os << "2"; break;
        case Rank::THREE: os << "3"; break;
        case Rank::FOUR:  os << "4"; break;
        case Rank::FIVE:  os << "5"; break;
        case Rank::SIX:   os << "6"; break;
        case Rank::SEVEN: os << "7"; break;
        case Rank::EIGHT: os << "8"; break;
        case Rank::NINE:  os << "9"; break;
        case Rank::TEN:   os << "T"; break;
        case Rank::JACK:  os << "J"; break;
        case Rank::QUEEN: os << "Q"; break;
        case Rank::KING:  os << "K"; break;
        case Rank::ACE:   os << "A"; break;
    }
    return os;
}

struct Card {
    Rank rank;
    Suit suit;

    Card(Rank r, Suit s) : rank(r), suit(s) {}

    int getValue() const;
    std::string toString() const;

    friend std::ostream& operator<<(std::ostream&, const Card&);
};

inline std::ostream& operator<<(std::ostream& os, const Card& card) {
    os << card.rank << card.suit;  // reuses the two overloads above
    return os;
}

// Example Deck skeleton (unchanged)
class Deck {
private:
    std::vector<Card> cards;
    void initialize();

public:
    Deck();
    const std::vector<Card>& getCards() const { return cards; }
    size_t size() const { return cards.size(); }
};
