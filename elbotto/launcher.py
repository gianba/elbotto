import logging
import argparse

from elbotto.bots import stochastic, rlagent

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',)

DEFAULT_SERVER_NAME = "ws://cs:3000"
# DEFAULT_SERVER_NAME = "ws://127.0.0.1:3000"
MODEL_SAVE_PATH = "../dqnagent/"
DEFAULT_ROUNDS_TO_PLAY = 20000

def launch_rl(bot_name, chosen_team_index, server_address=DEFAULT_SERVER_NAME, rounds_to_play=DEFAULT_ROUNDS_TO_PLAY,
           log=False, mode=rlagent.Mode.RUN):
    bot = rlagent.Bot(server_address,
                      bot_name,
                      chosen_team_index=chosen_team_index,
                      output_path=MODEL_SAVE_PATH,
                      rounds_to_play=rounds_to_play,
                      log=log,
                      mode=mode)
    bot.start()

def launch_stoc(bot_name, chosen_team_index, server_address=DEFAULT_SERVER_NAME, rounds_to_play=DEFAULT_ROUNDS_TO_PLAY):
    bot = stochastic.Bot(server_address,
                        name=bot_name,
                        chosen_team_index=chosen_team_index,
                        rounds_to_play=rounds_to_play)
    bot.start()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-n", "--name", required=True,
                    help="bot name")
    ap.add_argument("-rl", "--reinforcement_learning", required=False,
                    help="RL vs. random", default='True')
    ap.add_argument("-t", "--team_index", required=True, type=int,
                    help="team index")
    ap.add_argument("-r", "--rounds_to_play",
                    help="rounds to play", type=int, required=False, default=DEFAULT_ROUNDS_TO_PLAY)
    ap.add_argument("-l", "--log",
                    help="logging", required=False, default='False')
    ap.add_argument("-m", "--mode", required=True,
                    help="TRAIN or RUN mode")
    args = vars(ap.parse_args())

    if (args['reinforcement_learning'] == 'True'):
        launch_rl(args['name'], args['team_index'], rounds_to_play=args['rounds_to_play'], log=(args['log']=='True'), mode=rlagent.Mode[args['mode']])
    else:
        launch_stoc(args['name'], args['team_index'], rounds_to_play=args['rounds_to_play'])