# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:22 PM
# FileName: player.py

import api_pb2 as api
import numpy as np
from example.medusa.algorithm1.config import *


class Player(object):
    def __init__(self, game, q_learning, rng):
        self.game = game
        self.action_num = ACTION_NUM  # [0,1,2,..,action_num-1]
        self.q_learning = q_learning
        self.rng = rng
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_rate = (EPSILON_START - EPSILON_MIN) * 1.0 / EPSILON_DECAY
        self.move, self.swing, self.fire, self.apply = self.game.get_action_space()
        self.last_score = 0
        self.last_frame = 0

    def run_episode(self, epoch, replay_buffer, random_action=False, testing=False):
        episode_step = 0
        episode_reword = 0
        train_count = 0
        loss_sum = 0
        q_sum = 0.0
        q_count = 0
        need_start = True
        self.last_score = 0

        # do no operation steps.
        # max_no_op_steps = 10
        # for _ in range(self.rng.randint(5, max_no_op_steps)):
        #     st, _, _, _, _ = self.game.step(0)

        while True:
            while need_start:
                time.sleep(0.5)
                self.game.submit_reincarnation()
                time.sleep(0.5)
                _, state, _, _, _, _, _, _, _, _, _, _, _ = self.game.get_observation_with_info()
                if state == api.in_game:
                    need_start = False
            frame, \
            state, \
            location, \
            immutable_element, \
            mutable_element, \
            bodies, \
            asset_ownership, \
            self_asset, \
            self_status, \
            pointer, \
            score, \
            kill, \
            health = self.game.get_observation_with_info()
            if frame == self.last_frame or location is None or location.shape[2] == 0:
                time.sleep(0.01)
                continue
            self.last_frame = frame
            st = np.concatenate((location, immutable_element, mutable_element, bodies,
                                 asset_ownership, self_asset, self_status), axis=-1)
            action, max_q = self._choose_action(st, replay_buffer, testing, random_action)
            if max_q is not None:
                q_count += 1
                q_sum += max_q
            if action == 0:
                self.game.submit_action(frame, None, None, None, None)
            else:
                move = self.move[int(action-1)]
                self.game.submit_action(frame, move, None, None, None)
            reward = 0.1
            if score > self.last_score:
                reward += 1
                self.last_score = score
            if health == 0:
                reward -= 5
                need_start = True
                terminal = True
            else:
                terminal = False
            replay_buffer.add_sample(st, action, reward, terminal)
            episode_step += 1
            episode_reword += reward
            episode_score = score
            if terminal:
                break

            if not testing and episode_step % TRAIN_PER_STEP == 0 and not random_action:
                # print('-- train_policy_net episode_step=%d' % episode_step)
                imgs, actions, rs, terminal = replay_buffer.random_batch(32)
                loss = self.q_learning.train_policy_net(imgs, actions, rs, terminal)
                loss_sum += loss
                train_count += 1

        return episode_step, episode_reword, episode_score, loss_sum / (train_count + 0.0000001), q_sum / (
                    q_count + 0.0000001)

    def _choose_action(self, img, replay_buffer, testing, random_action):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)
        max_q = None
        random_num = self.rng.rand()

        if random_action or (not testing and random_num < self.epsilon):
            action = self.rng.randint(0, self.action_num)
        else:
            phi = replay_buffer.phi(img)
            action, max_q = self.q_learning.choose_action(phi, self.epsilon)
        return action, max_q



if __name__ == '__main__':
    pass
