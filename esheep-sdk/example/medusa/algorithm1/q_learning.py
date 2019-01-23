# Author: Taoz
# Date  : 8/26/2018
# Time  : 2:47 PM
# FileName: q_network.py

import numpy as np
from mxnet import init, nd, autograd, gluon
import example.medusa.algorithm1.dueling_dqn as dueling_dqn
import example.medusa.algorithm1.utils as g_utils
from example.medusa.algorithm1.config import *


class QLearning(object):
    def __init__(self, ctx, input_sample, model_file=None, is_dueling=False):
        self.ctx = ctx
        self.dueling = is_dueling
        self.policy_net = self.get_net(input_sample)
        self.target_net = self.get_net(input_sample)
        self.epsilon = EPSILON_START

        if model_file is not None:
            print('%s: read trained model from [%s]' % (time.strftime("%Y-%m-%d %H:%M:%S"), model_file))
            # self.policy_net.load_parameters(model_file, ctx=self.ctx)
            self.policy_net.load_params(model_file, ctx=self.ctx)

        self.update_target_net()

        # adagrad
        self.trainer = gluon.Trainer(self.policy_net.collect_params(), OPTIMIZER,
                                     {'learning_rate': LEARNING_RATE,
                                      'wd': WEIGHT_DECAY})
        self.loss_func = gluon.loss.L2Loss()

    def update_target_net(self):
        copy_params(self.policy_net, self.target_net)

    def choose_action(self, state, epsilon):
        shape0 = state.shape
        state = nd.array(state, ctx=self.ctx).reshape((1, -1, shape0[-2], shape0[-1])) / 255.0
        out = self.policy_net(state)
        self.epsilon = epsilon
        max_index = nd.argmax(out, axis=1)
        action = int(max_index.astype(np.int).asscalar())
        # print('state:', state)
        # print('state s:', state.shape)
        # print('out:', out)
        # print('out s:', out.shape)
        # print('max_index:', max_index)
        # print('max_index s:', max_index.shape)
        # print('action:', action)
        # print('action type:', type(action))
        max_q = out[0, action].asscalar()
        return action, max_q

    def train_policy_net(self, imgs, actions, rs, terminals):
        """
        Train one batch.

        Arguments:

        imgs - b x (f + 1) x C x H x W numpy array, where b is batch size,
               f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """
        batch_size = actions.shape[0]

        states = imgs[:, :-1, :, :, :]
        next_states = imgs[:, 1:, :, :, :]
        s = states.shape
        states = states.reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W
        next_states = next_states.reshape((s[0], -1, s[-2], s[-1]))  # batch x (f x C) x H x W

        st = nd.array(states, ctx=self.ctx, dtype=np.float32) / 255.0
        at = nd.array(actions[:, 0], ctx=self.ctx)
        rt = nd.array(rs[:, 0], ctx=self.ctx)
        tt = nd.array(terminals[:, 0], ctx=self.ctx)
        st1 = nd.array(next_states, ctx=self.ctx, dtype=np.float32) / 255.0

        if IS_DOUBLE:
            next_at = nd.argmax(self.policy_net(st1), axis=1)
            next_q_out = nd.pick(self.target_net(st1), next_at, 1)
        else:
            next_qs = self.target_net(st1)
            next_q_out = nd.max(next_qs, axis=1)

        target = rt + next_q_out * (1.0 - tt) * DISCOUNT

        with autograd.record():
            current_qs = self.policy_net(st)
            current_q = nd.pick(current_qs, at, 1)
            loss = self.loss_func(target, current_q)
            # diff = nd.abs(current_q - target)
            # quadratic_part = nd.clip(diff, -1, 1)
            # loss = 0.5 * nd.sum(nd.square(quadratic_part)) + nd.sum(diff - quadratic_part)

            # print('current_qs', current_qs)
            # print('current_q', current_q)
            # print('diff', diff)
            # print('quadratic_part', quadratic_part)
            # print('loss', loss)

        loss.backward()

        # 梯度裁剪
        if GRAD_CLIPPING_THETA is not None:
            params = [p.data() for p in self.policy_net.collect_params().values()]
            g_utils.grad_clipping(params, GRAD_CLIPPING_THETA, self.ctx)

        self.trainer.step(batch_size)
        total_loss = loss.mean().asscalar()
        return total_loss

    def q_vals(self, sample_batch):
        pass

    def save_params_to_file(self, model_path, mark):
        time_mark = time.strftime("%Y%m%d_%H%M%S")
        filename = model_path + '/net_' + str(mark) + '_' + time_mark + '.model'
        # self.policy_net.save_parameters(filename)
        self.policy_net.save_params(filename)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), ' save model success:', filename)

    def get_net(self, input_sample):
        if self.dueling:
            net = dueling_dqn.DuelingDQN()
            net.initialize(init.Xavier(), ctx=self.ctx)
        else:
            net = dueling_dqn.OriginDQN()
            net.initialize(init.Xavier(), ctx=self.ctx)

        net(input_sample)
        return net


def copy_params(src_net, dst_net):
    ps_src = src_net.collect_params()
    ps_dst = dst_net.collect_params()
    prefix_length = len(src_net.prefix)
    for k, v in ps_src.items():
        k = k[prefix_length:]
        v_dst = ps_dst.get(k)
        v_dst.set_data(v.data())

