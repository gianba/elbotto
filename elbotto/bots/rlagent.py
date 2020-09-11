import os
import logging
from enum import Enum
from collections import Counter

from tensorforce.agents import Agent

import numpy as np

from elbotto.basebot import BaseBot, DEFAULT_TRUMPF
from elbotto.card import Card, Color, CARD_OFFSET, CARDS_PER_COLOR
from elbotto.messages import GameType

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.WARNING)

class Mode(Enum):
    TRAIN = 0
    RUN = 1


REJECT_CARD_REWARD = -1
MAX_STICH = 9
SAVE_EPISODES = 1000


def get_states():
    # 4 * CARDS_PER_COLOR for all played cards
    # 4 * CARDS_PER_COLOR for cards on the table
    # 4 * CARDS_PER_COLOR for the current hand
    # 4 currently played color
    # 3 * 4 states for "potentially still has trump / other colors"
    # 4 states for "who starts"
    # 2 states for "who owns current stich"
    return dict(
        cards=dict(type='float', shape=(4*CARDS_PER_COLOR, 3)),
        features=dict(type='float', shape=5 * 4 + 2)
    )

def get_actions():
    return dict(type='int', shape=(), num_values=4 * CARDS_PER_COLOR)

class Bot(BaseBot):
    """
    Trivial bot using DEFAULT_TRUMPF and randomly returning a card available in the hand.
    This is a simple port of the original Java Script implementation
    """

    def __init__(self, server_address, name, chosen_team_index=0, output_path=None, rounds_to_play=1, log=False,
                 mode=Mode.TRAIN):
        super(Bot, self).__init__(server_address, name, chosen_team_index, rounds_to_play)

        self.mode = mode
        self.episode = 1
        self.stich_number = 0
        self.played_cards_in_game = []
        self.rejected_cards = []

        self.out_of_color = np.zeros((3, 4), dtype=np.float)

        self.avg_stich_reward = 0
        self.rejected_per_session = 0
        self.avg_rejected_per_session = 0

        if log:
            self.log_game(output_path)
        model_path = os.path.join(output_path, 'checkpoints')

        if mode is Mode.TRAIN:
            os.makedirs(output_path, exist_ok=True)
            if os.path.exists(model_path):
                self.agent = Agent.load(model_path)
            else:
                self.agent = Agent.create(agent='dqn',
                                          states=get_states(),
                                          actions=get_actions(),
                                          max_episode_timesteps=50,
                                          memory=20000,
                                          batch_size=32,
                                          start_updating=8192,
                                          exploration=0.1,
                                          network=[
                                              [dict(type='retrieve', tensors=['cards']),
                                               dict(type='conv1d', size=256, window=9, stride=9, padding='valid'),
                                               dict(type='flatten'),
                                               dict(type='register', tensor='cards-embedding')],
                                              [dict(type='retrieve', tensors=['features']),
                                               dict(type='dense', size=64, activation='relu'),
                                               dict(type='register', tensor='features-embedding')],
                                              [dict(type='retrieve', aggregation='concat',
                                                    tensors=['cards-embedding', 'features-embedding']
                                               ),
                                               dict(type='dense', size=512, activation='relu'),
                                               dict(type='dense', size=256, activation='relu'),
                                               dict(type='dense', size=256, activation='relu')]
                                            # dict(type='dense', size=256, activation='tanh'),
                                            # dict(type='dense', size=256, activation='tanh'),
                                            # dict(type='dense', size=128, activation='tanh'),
                                            # dict(type='dense', size=128, activation='tanh')
                                            # dict(type='dense', size=512, activation='relu'),
                                            # dict(type='dropout', rate=0.2),
                                            # dict(type='dense', size=512, activation='relu'),
                                            # dict(type='dropout', rate=0.2),
                                            # dict(type='dense', size=256, activation='relu'),
                                            # dict(type='dropout', rate=0.2),
                                            # dict(type='dense', size=256, activation='relu'),
                                            # dict(type='dense', size=256, activation='relu')
                                          ],
                                          discount=1.0,
                                          summarizer=dict(
                                            directory=os.path.join(output_path, "summary"),
                                            labels=['entropy', 'kl-divergence', 'loss', 'reward', 'update-norm']
                                          ),
                                          saver=dict(
                                            directory=os.path.join(output_path, "checkpoints"),
                                            frequency=SAVE_EPISODES  # save checkpoint every 100 updates
                                          )
                )
        else:
            self.agent = Agent.load(model_path)


    def handle_request_trumpf(self):
        cnt = Counter()
        for card in self.handCards:
            cnt[card.color] += 1
        most_common_color = cnt.most_common(1)[0][0]
        return GameType("TRUMPF", most_common_color.name)

    def handle_reject_card(self, card):
        if self.mode is Mode.TRAIN:
            if self.log:
                logger.warning('Reject reward {} for bot {} due to card {}'.format(REJECT_CARD_REWARD, self.name, card))
            self.agent.observe(reward=REJECT_CARD_REWARD, terminal=False)

        self.rejected_cards.append(card)
        self.rejected_per_session += 1

    def handle_request_card(self, tableCards):
        state = self._build_state(tableCards)
        exploit = (self.mode is Mode.RUN)
        action = self.agent.act(states=state, deterministic=exploit, independent=exploit)
        card = self._convert_action_to_card(action)

        return card

    def _convert_action_to_card(self, action):
        card = Card.from_idx(int(action), self.game_type.trumpf_color)
        for hard_card in self.handCards:
            if card == hard_card:
                return card

        return None

    def handle_game_finished(self):
        self.episode += 1
        self.stich_number = 0
        self.played_cards_in_game = []
        self.out_of_color = np.zeros((3, 4), dtype=np.float)

    def handle_played_cards(self, played_cards):
        super(Bot, self).handle_played_cards(played_cards)

        for card in played_cards:
            if card not in self.played_cards_in_game:
                self.played_cards_in_game.append(card)

    def handle_stich(self, winner, round_points, total_points, played_cards, seat_id):
        self.rejected_cards = []
        self.stich_number += 1
        if self.in_my_team(winner):
            reward = round_points / (self.get_trumpf_factor() * 56)
        else:
            reward = -round_points / (self.get_trumpf_factor() * 56)
        if self.mode is Mode.TRAIN:
            self.agent.observe(reward=reward, terminal=self.stich_number==MAX_STICH)
        self.avg_stich_reward = 0.9999 * self.avg_stich_reward + 0.0001 * reward

        self.check_out_of_color(played_cards, seat_id)

    def check_out_of_color(self, played_cards, seat_id):
        stich_color_id = Card.id_from_color(played_cards[0].color)
        for i, seat in enumerate(range(seat_id+1, seat_id+4)):  # iterate over all seats
            seat_cor = (seat - self.player.seatId) % 4
            if seat_cor != 0:     # don't track out-of-color the player himself
                if played_cards[0].color is not played_cards[i].color and played_cards[i].color is not self.game_type.trumpf_color:
                    self.out_of_color[seat_cor-1, stich_color_id] = 1

    def handle_winner_team(self, data):
        self.avg_rejected_per_session = 0.99 * self.avg_rejected_per_session + 0.01 * self.rejected_per_session
        logger.info('Average stich reward: {:.5f} and {} (avg: {:.2f}) rejected cards in this session'.format(self.avg_stich_reward, self.rejected_per_session, self.avg_rejected_per_session))
        self.rejected_per_session = 0

    def _build_state(self, tableCards):
        order = {}
        for color in Color:
            order[color] = (color.value - self.game_type.trumpf_color.value) % 4

        state = np.zeros((4*CARDS_PER_COLOR, 3), dtype=np.float)
        action_mask = np.zeros((4, CARDS_PER_COLOR), dtype=np.float)

        # already played cards (36 states)
        for card in self.played_cards_in_game:
            state[order[card.color] * CARDS_PER_COLOR + card.number - CARD_OFFSET, 0] = 1.0

        # cards currently on the table (36 states)
        for card in tableCards:
            state[order[card.color] * CARDS_PER_COLOR + card.number - CARD_OFFSET, 1] = 1.0

        # cards in player hand (36 states)
        for card in self.handCards:
            state[order[card.color] * CARDS_PER_COLOR + card.number - CARD_OFFSET, 2] = 1.0
            if card not in self.rejected_cards:
                action_mask[order[card.color], card.number - CARD_OFFSET] = 1.0

        # currently played color (4 states)
        played_color = np.zeros((4, 2), dtype=np.float)
        if len(tableCards) >= 1:
            played_color[order[tableCards[0].color], 0] = 1.0
        # player who started stich (4 states)
        played_color[len(tableCards), 1] = 1.0

        # team who currently owns the stich (2 states)
        stich_owner = self.get_stich_owner(tableCards)

        # return dict(state=dict(cards=state,
        #                        features=np.concatenate((played_color.flatten(),
        #                                                 self.out_of_color.flatten(),
        #                                                 stich_owner))),
        #             action_mask=action_mask.flatten())

        return dict(cards=state,
                    features=np.concatenate((played_color.flatten(),
                                                        self.out_of_color.flatten(),
                                                        stich_owner)),
                    action_mask=action_mask.flatten())

    def get_stich_owner(self, tableCards):
        stich_owner = np.zeros(2, dtype=np.float)
        if len(tableCards) > 0:
            winning_color = tableCards[0].color
            for card in tableCards:
                if card.color is self.game_type.trumpf_color:
                    winning_color = card.color

            player_idx = 0
            card_number = 0
            for i, card in enumerate(tableCards):
                if card.color is winning_color:
                    if card.number > card_number:
                        player_idx = i

            stich_owner[(len(tableCards) - player_idx) % 2] = 1
        return stich_owner

    def get_trumpf_factor(self):
        factor = 1 if self.game_type.trumpf_color in [Color.HEARTS, Color.DIAMONDS] else 2
        return factor



