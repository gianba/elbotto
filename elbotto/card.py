from enum import Enum


class Color(Enum):
    HEARTS= 0
    DIAMONDS= 1
    CLUBS= 2
    SPADES= 3


CARD_OFFSET = 6
CARDS_PER_COLOR = 9

class Card(object):

    def __init__(self, number, color):
        self.number = number
        self.color = Color[color]
        self.id = (self.number - CARD_OFFSET) + (self.color.value * CARDS_PER_COLOR)

    @staticmethod
    def create(number, color):
        return Card(number, color)

    @staticmethod
    def from_idx(idx, trumpf_color=Color.HEARTS):
        color_idx = idx // 9
        color_idx_trumpf_corrected = (color_idx + trumpf_color.value) % 4
        number = idx - (color_idx * 9) + CARD_OFFSET
        color = Color(color_idx_trumpf_corrected)
        return Card.create(number, color.name)

    @staticmethod
    def id_from_color(color, trumpf_color=Color.HEARTS):
        return (color.value - trumpf_color.value) % 4

    def to_dict(self):
        return dict(number=self.number,
                    color=self.color.name)

    def __eq__(self, other):
        return self.number == other.number and self.color == other.color

    def __repr__(self):
        return "%s - %s"%(self.number, self.color)