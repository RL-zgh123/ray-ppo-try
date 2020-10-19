import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import ray
from ray.experimental.tf_utils import TensorFlowVariables
import time

N_WORKERS = 1
ITERATIONS = 250000
N_TEST = 10
EP_MAX = 200
EP_LEN = 200
GAMMA = 0.9
A_LR = 1e-4
C_LR = 5e-4
BATCH = 64
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
GAME = 'Pendulum-v0'
S_DIM, A_DIM = 3, 1
METHOD = dict(name='clip', epsilon=0.2)


class PPO(object):
    def __init__(self):
        # build placeholder
        self._build_ph('ph')

        # init weight
        self.w_init, self.b_init = tf.random_normal_initializer(0, 0.3), \
                                   tf.constant_initializer(0.1)
        # build critic net
        self._build_cnet('critic')

        # build actor net
        self.pi, self.pi_params, self.sigma = self._build_anet('pi', trainable=True)
        self.oldpi, self.oldpi_params, _ = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(self.pi.sample(1), axis=0)

        # build loss function
        self._build_loss_function('loss')

        # update oldpi by pi
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in
                                    zip(self.pi_params, self.oldpi_params)]

        # init session
        self.sess = tf.Session()
        tf.summary.FileWriter("log/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        # get variables by ray method
        self._build_variables()

    def _build_ph(self, name):
        with tf.variable_scope(name):
            self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
            self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu,
                                 kernel_initializer=self.w_init,
                                 bias_initializer=self.b_init,
                                 trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh,
                                     kernel_initializer=self.w_init,
                                     bias_initializer=self.b_init,
                                     trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus,
                                    kernel_initializer=self.w_init,
                                    bias_initializer=self.b_init,
                                    trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params, sigma

    def _build_cnet(self, name):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu,
                                 kernel_initializer=self.w_init,
                                 bias_initializer=self.b_init
                                 )
            self.v = tf.layers.dense(l1, 1,
                                     kernel_initializer=self.w_init,
                                     bias_initializer=self.b_init
                                     )  # state-value
            self.advantage = self.tfdc_r - self.v

    def _build_loss_function(self, name):
        with tf.variable_scope('name'):
            with tf.variable_scope('atrain'):
                a0 = tf.constant(1e-10)
                self.ratio = self.pi.prob(self.tfa) / (
                    self.oldpi.prob(self.tfa) + a0)
                self.surr = self.ratio * self.tfadv
                self.aloss = -tf.reduce_mean(
                    tf.minimum(self.surr,
                               tf.clip_by_value(self.ratio, 1. - METHOD['epsilon'],
                                                1. + METHOD[
                                                    'epsilon']) * self.tfadv))
                self.actor_opt = tf.train.AdamOptimizer(A_LR)
                self.atrain_op = self.actor_opt.minimize(
                    self.aloss)

            with tf.variable_scope('ctrain'):
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                self.critic_opt = tf.train.AdamOptimizer(C_LR)
                self.ctrain_op = self.critic_opt.minimize(self.closs)

    def _build_variables(self):
        # ray专用的权重提取api
        self.a_variables = ray.experimental.tf_utils.TensorFlowVariables(
            self.aloss, self.sess)
        self.c_variables = ray.experimental.tf_utils.TensorFlowVariables(
            self.closs, self.sess)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a, sigma = self.sess.run([self.sample_op, self.sigma], {self.tfs: s})
        return np.clip(a[0], -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def update(self, datas):
        transitions, bs_ = datas[0], datas[1]
        for i, transition in enumerate(transitions):
            if len(transition) > 10:
                self.sess.run(self.update_oldpi_op)
                # compute discounted reward
                # br = transition[:, -1:]
                # # print('br:', br)
                # v_s_ = self.get_v(bs_[i])
                # discount_r = []
                # for r in br[::-1][0]:
                #     v_s_ = r + GAMMA * v_s_
                #     discount_r.append(v_s_)
                # discount_r.reverse()
                # br_ = np.array(discount_r)[:, np.newaxis]
                # bs, ba= transition[:, :S_DIM], transition[:, S_DIM: S_DIM + A_DIM]

                bs, ba, br = transition[:, :S_DIM], transition[:, S_DIM: S_DIM + A_DIM], transition[:, -1:]

                adv = self.sess.run(self.advantage, {self.tfs: bs, self.tfdc_r: br})
                # udpate actor and critic in a loop
                [self.sess.run(self.atrain_op,
                               {self.tfs: bs, self.tfa: ba, self.tfadv: adv}) for _ in
                 range(A_UPDATE_STEPS)]
                [self.sess.run(self.ctrain_op, {self.tfs: bs, self.tfdc_r: br}) for _ in
                 range(C_UPDATE_STEPS)]
        return self.sess.run([self.aloss, self.closs],
                                 {self.tfs: bs, self.tfa: ba, self.tfadv: adv,
                                  self.tfdc_r: br})

    def get_weights(self):
        return [self.a_variables.get_weights(), self.c_variables.get_weights()]

    def set_weights(self, weights):
        self.a_variables.set_weights(weights[0])
        self.c_variables.set_weights(weights[1])


# PS can be deployed to GPU machine
@ray.remote
class ParameterServer(object):
    def __init__(self):
        self.global_ppo = PPO()

    def update_model(self, data):
        # transitions = ray.get(transitions)
        aloss, closs = self.global_ppo.update(data)
        return self.global_ppo.get_weights()

    def get_weights(self):
        return self.global_ppo.get_weights()

    def get_action(self, state):
        return self.global_ppo.choose_action(state)


@ray.remote
class DataWorker(object):
    def __init__(self):
        self.local_ppo = PPO()
        self.env = gym.make(GAME).unwrapped

    # output data (receive new parameters from PS)
    def compute_transitions(self, weights):
        self.local_ppo.set_weights(weights)
        buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []
        ep_r = 0
        transitions = []
        s = self.env.reset()

        for i in range(EP_LEN):
            a = self.local_ppo.choose_action(s)
            s_, r, done, _ = self.env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8)/8)
            s = s_
            ep_r += r

            if i % BATCH == 0 or i == EP_LEN - 1:
                # compute discounted reward
                v_s_ = self.local_ppo.get_v(s_)
                discount_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discount_r.append(v_s_)
                discount_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(
                    discount_r)[:, np.newaxis]

                # for computing dc_r in ps
                # buffer_s_.append(s_)
                # bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_r)


                transtion = np.hstack((bs, ba, br))
                transitions.append(transtion)

                buffer_s, buffer_a, buffer_r = [], [], []

        # print(ep_r)
        return transitions, buffer_s_

def main():
    """
    define the way of data interaction among workers and parametersever
    """
    print("Running Asynchronous Parameter Server Training.")

    ray.init()
    ps = ParameterServer.remote()
    workers = [DataWorker.remote() for _ in range(N_WORKERS)]

    current_weights = ps.get_weights.remote()
    datas = {}
    for worker in workers:
        datas[worker.compute_transitions.remote(current_weights)] = worker

    with tf.Graph().as_default():
        test_ppo = PPO()
    test_env = gym.make(GAME).unwrapped
    test_count = 0

    for i in range(ITERATIONS * N_WORKERS):
        ready_list, _ = ray.wait(list(datas))
        ready_id = ready_list[0]
        worker = datas.pop(ready_id)

        # update PS with worker transitions
        current_weights = ps.update_model.remote(ready_id)
        datas[worker.compute_transitions.remote(current_weights)] = worker

        # evalute temporal performance
        if i % (250 * N_WORKERS) == 0:
            test_count += 1
            ep_r = 0
            weights = ray.get(current_weights)
            test_ppo.set_weights(weights)

            for i in range(N_TEST):
                s = test_env.reset()
                r0 = ep_r
                for _ in range(200):
                    a = test_ppo.choose_action(s)
                    s_, r, done, _ = test_env.step(a)
                    ep_r += r
                    s = s_
            print('{} round, {} tests, {} points'.format(test_count, N_TEST,
                                                         np.round(ep_r / N_TEST, 2)))


if __name__ == "__main__":
    main()
