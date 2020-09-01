import logging
import argparse

from elbotto.bots import stochastic, rlagent

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',)

DEFAULT_SERVER_NAME = "ws://127.0.0.1:3000"
MODEL_SAVE_PATH = "../dqnagent/"
DEFAULT_ROUNDS_TO_PLAY = 10000

def launch(bot_name, chosen_team_index, server_address=DEFAULT_SERVER_NAME, rounds_to_play=DEFAULT_ROUNDS_TO_PLAY,
           log=False, mode=rlagent.Mode.RUN):
    bot = rlagent.Bot(server_address,
                      bot_name,
                      chosen_team_index=chosen_team_index,
                      output_path=MODEL_SAVE_PATH+bot_name,
                      rounds_to_play=rounds_to_play,
                      log=log,
                      mode=mode)
    bot.start()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("-n", "--name", required=True,
                    help="bot name")
    ap.add_argument("-t", "--team_index", required=True,
                    help="team index")
    ap.add_argument("-r", "--rounds_to_play", required=False,
                    help="rounds to play")
    ap.add_argument("-l", "--log", required=False,
                    help="logging")
    ap.add_argument("-m", "--mode", required=True,
                    help="TRAIN or RUN mode")
    args = vars(ap.parse_args())

    launch(args['name'], int(args['team-index']), rounds_to_play=int(args['rounds_to_play']), log=bool(args['log']), mode=rlagent.Mode(args['mode']))