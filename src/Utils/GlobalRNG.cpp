#include "GlobalRNG.h"
#include "Shoe/Shoe.h"
#include <algorithm>

void GlobalRNG::shuffleShoe(Shoe& shoe) {
    auto& cards = shoe.getCards();
    int n = static_cast<int>(cards.size());
    for (int i = n - 1; i >= 1; --i) {
        int j = nextInt(i + 1);
        std::swap(cards[i], cards[j]);
    }
    shoe.resetIndex();
}
