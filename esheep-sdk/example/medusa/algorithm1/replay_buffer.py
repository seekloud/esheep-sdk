# Author: Taoz
# Date  : 8/26/2018
# Time  : 11:52 AM
# FileName: replay_buffer.py

# copy from https://github.com/zmonoid/mxdqn/blob/master/ale_data_set.py
# edit a little.


"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""
import numpy as np

floatX = 'float32'

from example.medusa.algorithm1.config import PHI_LENGTH


class ReplayBuffer(object):
    """A replay memory consisting of circular buffers for observed images,
actions, and rewards.

    """

    def __init__(self, height, width, channel, rng, max_steps=10000):
        """Construct a DataSet.

        Arguments:
            width, height, channel - image size
            max_steps - the number of time steps to store
            phi_length - number of images to concatenate into a state
            rng - initialized numpy random number generator, used to
            choose random minibatches

        """
        # TODO: Specify capacity in number of state transitions, not number of saved time steps.

        # Store arguments.
        self.width = width
        self.height = height
        self.channel = channel
        self.max_steps = max_steps
        self.phi_length = PHI_LENGTH
        self.rng = rng

        # Allocate the circular buffers and indices.
        self.imgs = np.zeros((max_steps, channel, height, width), dtype='uint8')
        self.actions = np.zeros(max_steps, dtype='int32')
        self.rewards = np.zeros(max_steps, dtype=floatX)
        self.terminal = np.zeros(max_steps, dtype='bool')
        self.terminal[-1] = True  # set terminal for the first episode.

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, img, action, reward, terminal):
        """Add a time step record.
        Arguments:
            img -- observed image (height, width, channel), it will be changed to (channel, height, width)
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.imgs[self.top] = img.transpose(2, 0, 1)
        self.actions[self.top] = action
        # self.rewards[self.top] = max(1, min(-1, reward))  # clip reward
        self.rewards[self.top] = reward  # clip reward
        self.terminal[self.top] = terminal

        if terminal:
            idx = self.top
            while True:
                idx -= 1
                if self.terminal[idx]:
                    break

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - self.phi_length)

    def phi(self, img):
        """Return a phi (sequence of image frames), using the last phi_length -
        1, plus img.

        """
        img = img.transpose(2, 0, 1)
        indexes = np.arange(self.top - self.phi_length + 1, self.top)

        phi = np.empty((self.phi_length, self.channel, self.height, self.width), dtype=floatX)
        phi[0:self.phi_length - 1] = self.imgs.take(indexes, axis=0, mode='wrap')

        phi[-1] = img
        return phi

    def random_batch(self, batch_size):
        """Return corresponding imgs, actions, rewards, and terminal status for
batch_size randomly chosen state transitions.

        """
        # Allocate the response.
        imgs = np.zeros((batch_size,
                         self.phi_length + 1,
                         self.channel,
                         self.height,
                         self.width),
                        dtype='uint8')
        actions = np.zeros((batch_size, 1), dtype='int32')
        rewards = np.zeros((batch_size, 1), dtype=floatX)
        terminal = np.zeros((batch_size, 1), dtype='bool')

        count = 0
        while count < batch_size:
            # print('count:', count)
            # print('batch_size:', batch_size)
            # print('self.bottom:', self.bottom)
            # print('self.size:', self.size)
            # print('self.phi_length:', self.phi_length)

            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - self.phi_length)

            # Both the before and after states contain phi_length
            # frames, overlapping except for the first and last.
            all_indices = np.arange(index, index + self.phi_length + 1)
            end_index = index + self.phi_length - 1

            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but its last frame (the
            # second-to-last frame in imgs) may be terminal. If the last
            # frame of the initial state is terminal, then the last
            # frame of the transitioned state will actually be the first
            # frame of a new episode, which the Q learner recognizes and
            # handles correctly during training by zeroing the
            # discounted future reward estimate.
            if np.any(self.terminal.take(all_indices[0:-2], mode='wrap')):
                continue

            # Add the state transition to the response.
            imgs[count] = self.imgs.take(all_indices, axis=0, mode='wrap')
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminal.take(end_index, mode='wrap')
            count += 1

        return imgs, actions, rewards, terminal


def test1():
    rng = np.random.RandomState()
    buff = ReplayBuffer(5, 3, rng, 20)

    for i in range(12):
        img = np.arange(i, i + 15, dtype=np.uint8).reshape(3, 5)
        buff.add_sample(img, i, i + 0.1, False)

    img = np.arange(100, 100 + 15, dtype=np.uint8).reshape(3, 5)
    buff.add_sample(img, 100, 100 + 0.1, True)

    imgs, actions, rewards, terminal = buff.random_batch(2)

    print('-------------------')
    print(buff.imgs)
    print('-------------------')
    print(buff.actions)
    print('-------------------')
    print(buff.rewards)
    print('-------------------')
    print(buff.terminal)

    print('-------------------')
    print('-------------------')
    print('-------------------')
    print('-------------------')
    print(imgs)
    print('-------------------')
    print(actions)
    print('-------------------')
    print(rewards)
    print('-------------------')
    print(terminal)


if __name__ == '__main__':
    test1()
