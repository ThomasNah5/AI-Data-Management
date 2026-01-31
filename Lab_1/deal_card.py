import random


def deal(num_hands, cards_per_hand, deck):
    hands = [[] for _ in range(num_hands)]

    for _ in range(cards_per_hand):
        for player in range(num_hands):
            card = deck.pop(0)
            hands[player].append(card)

    return hands



class cards():
    cards = []

    def create(self):
        self.cards = []
        values = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        suits = ["s", "h", "d", "c"]

        for value in values:
            for suit in suits:
                self.cards.append(value + suit)

    def shuffle(self):
        for i in range(len(self.cards)):
            random_index = random.randint(0, len(self.cards) - 1)
            self.cards[i], self.cards[random_index] = self.cards[random_index], self.cards[i]

    def deal(self, hands, card_num):
        dealt_hands = [[] for _ in range(hands)]

        for _ in range(card_num):
            for player in range(hands):
                card = self.cards.pop(0)
                dealt_hands[player].append(card)

        return dealt_hands



def main():
    card_01 = cards()
    card_01.create()

    print("Deck after creating:")
    print(card_01.cards)
    print(f"Total cards: {len(card_01.cards)}\n")

    card_01.shuffle()

    print("Deck after shuffling:")
    print(card_01.cards)
    print(f"Total cards: {len(card_01.cards)}\n")

    num_hands = 4
    cards_per_hand = 5
    dealt = card_01.deal(num_hands, cards_per_hand)

    print(f"Dealing {cards_per_hand} cards to {num_hands} players:\n")
    for i, hand in enumerate(dealt):
        print(f"  Player {i + 1}: {hand}")

    print(f"\nRemaining deck: {card_01.cards}")
    print(f"Cards remaining: {len(card_01.cards)}")

main()
