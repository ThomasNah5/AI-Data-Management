import random


def createDeck():
    values = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["s", "h", "d", "c"]
    deck = []

    for value in values:
        for suit in suits:
            deck.append(value + suit)

    return deck


def shuffle(deck):
    before = deck.copy()

    for i in range(len(deck)):
        random_index = random.randint(0, len(deck) - 1)
        deck[i], deck[random_index] = deck[random_index], deck[i]

    after = deck

    return before, after


# Main program
def main():
    deck = createDeck()
    before, after = shuffle(deck)

    print("Before shuffling:")
    print(before)
    print(f"\nAfter shuffling:")
    print(after)


main()
