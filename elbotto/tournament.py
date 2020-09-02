import logging
import threading

from elbotto.bots import stochastic, rlagent

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG)

DEFAULT_BOT_NAME = "El botto del jasso"

DEFAULT_SERVER_NAME = "ws://127.0.0.1:3000"

OUTPUT_PATH = "../dqnagent/"

ROUNDS_TO_PLAY = 100000

def create_bot_thread(bot):
    return threading.Thread(target=bot.start)


def start_bots():
    # create_bot_thread(rlagent.Bot(DEFAULT_SERVER_NAME,
    #                               name="El botto del jasso 0",
    #                               chosen_team_index=0,
    #                               output_path=OUTPUT_PATH + "bot0/",
    #                               rounds_to_play=ROUNDS_TO_PLAY,
    #                               log=True,
    #                               mode=rlagent.Mode.NEW)).start()
    #
    # create_bot_thread(rlagent.Bot(DEFAULT_SERVER_NAME,
    #                               name="El botto del jasso 1",
    #                               chosen_team_index=0,
    #                               output_path=OUTPUT_PATH + "bot1/",
    #                               rounds_to_play=ROUNDS_TO_PLAY,
    #                               mode=rlagent.Mode.NEW)).start()


    create_bot_thread(rlagent.Bot(DEFAULT_SERVER_NAME,
                                  name="El botto del jasso 0",
                                  chosen_team_index=0,
                                  output_path=OUTPUT_PATH,
                                  rounds_to_play=ROUNDS_TO_PLAY,
                                  log=False,
                                  mode=rlagent.Mode.TRAIN)).start()

    create_bot_thread(rlagent.Bot(DEFAULT_SERVER_NAME,
                                  name="El botto del jasso 1",
                                  chosen_team_index=0,
                                  output_path=OUTPUT_PATH,
                                  rounds_to_play=ROUNDS_TO_PLAY,
                                  mode=rlagent.Mode.TRAIN)).start()

    # create_bot_thread(rlagent.Bot(DEFAULT_SERVER_NAME,
    #                               name="El botto del jasso 2",
    #                               chosen_team_index=1,
    #                               output_path=OUTPUT_PATH + "bot2/",
    #                               rounds_to_play=ROUNDS_TO_PLAY,
    #                               mode=rlagent.Mode.NEW)).start()
    #
    # t = create_bot_thread(rlagent.Bot(DEFAULT_SERVER_NAME,
    #                               name="El botto del jasso 3",
    #                               chosen_team_index=1,
    #                               output_path=OUTPUT_PATH + "bot3",
    #                               rounds_to_play=ROUNDS_TO_PLAY,
    #                               mode=rlagent.Mode.NEW))

    create_bot_thread(stochastic.Bot(DEFAULT_SERVER_NAME,
                                     name="stochastic 0",
                                     chosen_team_index=1,
                                     rounds_to_play=ROUNDS_TO_PLAY)).start()

    t = create_bot_thread(stochastic.Bot(DEFAULT_SERVER_NAME,
                                     name="stochastic 1",
                                     chosen_team_index=1,
                                     rounds_to_play=ROUNDS_TO_PLAY))


    t.start()
    t.join()


if __name__ == '__main__':
    start_bots()