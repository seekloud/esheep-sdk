# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:13 PM
# FileName: experiment.py


import numpy as np
from example.medusa.algorithm1.player import Player
from game_env import GameEnvironment
from example.medusa.algorithm1.replay_buffer import ReplayBuffer
from example.medusa.algorithm1.q_learning import QLearning
from example.medusa.algorithm1 import utils
import mxnet as mx
from mxnet import nd
import example.medusa.algorithm1.ztutils as ztutils
from example.medusa.algorithm1.config import *


class Experiment(object):
    ctx = utils.try_gpu(GPU_INDEX)
    mx.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    INPUT_SAMPLE = nd.random.uniform(0, 255, (1, PHI_LENGTH * CHANNEL, HEIGHT, WIDTH), ctx=ctx) / 255.0

    def __init__(self, testing=False):
        ztutils.mkdir_if_not_exist(MODEL_PATH)
        self.step_count = 0
        self.episode_count = 0
        self.target_net_update_count = 0
        self.q_learning = QLearning(Experiment.ctx,
                                    Experiment.INPUT_SAMPLE,
                                    model_file=PRE_TRAIN_MODEL_FILE,
                                    is_dueling=IS_DUELING,
                                    )
        ip = '127.0.0.1'
        port = '5321'
        self.game = GameEnvironment(ip=ip, port=port, api_token="test")

        self.player = Player(self.game,
                             self.q_learning,
                             Experiment.rng)

        self.replay_buffer = ReplayBuffer(HEIGHT,
                                          WIDTH,
                                          CHANNEL,
                                          Experiment.rng,
                                          BUFFER_MAX)
        self.update_target_episode = UPDATE_TARGET_BY_EPISODE_BEGIN
        self.update_target_interval = UPDATE_TARGET_BY_EPISODE_BEGIN + UPDATE_TARGET_RATE
        self.testing = testing

    def start_train(self):
        self.game.create_room("123")
        for i in range(1, EPOCH_NUM + 1):
            self._run_epoch(i)
        print('train done.')

    def start_test(self):
        # assert PRE_TRAIN_MODEL_FILE is not None
        self.game.create_room("123")
        for i in range(1, EPOCH_NUM + 1):
            self._run_epoch(i)
        print('test done.')

    def _run_epoch(self, epoch):
        steps_left = EPOCH_LENGTH
        random_episode = True
        episode_in_epoch = 0
        step_in_epoch = 0
        reward_in_epoch = 0.0
        score_in_epoch = 0.0
        while steps_left > 0:
            if self.step_count > BEGIN_RANDOM_STEP:
                random_episode = False
            t0 = time.time()
            ep_steps, \
            ep_reward, \
            ep_score, \
            avg_loss, \
            avg_max_q = self.player.run_episode(epoch,
                                                self.replay_buffer,
                                                random_action=random_episode,
                                                testing=self.testing)

            self.step_count += ep_steps
            if not random_episode:
                self.episode_count += 1
                episode_in_epoch += 1
                score_in_epoch += ep_score
                step_in_epoch += ep_steps
                reward_in_epoch += ep_reward
                steps_left -= ep_steps
            t1 = time.time()

            print(
                'episode [%d], episode step=%d, total_step=%d, time=%.2fs, score=%.2f, ep_reward=%.2f, avg_loss=%.4f, avg_q=%f'
                % (self.episode_count, ep_steps, self.step_count, (t1 - t0), ep_score, ep_reward, avg_loss, avg_max_q))
            print('')
            self._update_target_net(random_episode)

        self._save_net()
        print('\n%s EPOCH finish [%d], episode=%d, step=%d, avg_step=%d, avg_score=%.2f avg_reward=%.2f \n\n\n' %
              (time.strftime("%Y-%m-%d %H:%M:%S"),
               epoch,
               self.episode_count,
               self.step_count,
               step_in_epoch // episode_in_epoch,
               score_in_epoch / episode_in_epoch,
               reward_in_epoch / episode_in_epoch))

    def _update_target_net(self, random_action=False):
        if not self.testing and self.episode_count == self.update_target_episode and not random_action:
            self.target_net_update_count += 1
            print('%s UPDATE TARGET NET, interval[%.3f], update count[%d]\n' % (
                time.strftime("%Y-%m-%d %H:%M:%S"), self.update_target_interval, self.target_net_update_count))

            self.update_target_episode = int(self.update_target_episode + self.update_target_interval)
            self.update_target_interval = min((self.update_target_interval + UPDATE_TARGET_RATE),
                                              UPDATE_TARGET_BY_EPISODE_END)

            self.q_learning.update_target_net()

    def _save_net(self):
        if not self.testing:
            self.q_learning.save_params_to_file(MODEL_PATH, MODEL_FILE_MARK + BEGIN_TIME)


def train():
    print(' ====================== START TRAIN ========================')
    exper = Experiment()
    exper.start_train()


def test():
    print(' ====================== START test ========================')
    exper = Experiment(testing=True)
    exper.start_test()


def test_speed():
    pass


if __name__ == '__main__':
    train()
