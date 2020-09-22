import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import ray
from ray.experimental.tf_utils import TensorFlowVariables

ray.init()

CORE_NUM = 10
EP_MAX = 200
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
    dict(name='clip', epsilon=0.2),  # Clipped surrogate objective
][1]


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)  # state-value
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.critic_opt = tf.train.AdamOptimizer(C_LR)
            self.ctrain_op = self.critic_opt.minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in
                                    zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv

            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -tf.reduce_mean(surr - self.tflam * kl)

            else:
                self.aloss = -tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(ratio, 1. - METHOD['epsilon'],
                                                      1. + METHOD[
                                                          'epsilon']) * self.tfadv))

            with tf.variable_scope('atrain'):
                self.actor_opt = tf.train.AdamOptimizer(A_LR)
                self.atrain_op = self.actor_opt.minimize(
                    self.aloss)

            tf.summary.FileWriter("log/", self.sess.graph)

            # ray专用的权重提取api
            self.a_variables = ray.experimental.tf_utils.TensorFlowVariables(
                self.aloss, self.sess)
            self.c_variables = ray.experimental.tf_utils.TensorFlowVariables(
                self.closs, self.sess)

            self.sess.run(tf.global_variables_initializer())

    def update0(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage,
                            {self.tfs: s, self.tfdc_r: r})  # 得到advantage value

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run([self.atrain_op, self.kl_mean],
                                      {self.tfs: s, self.tfa: a,
                                       self.tfadv: adv,
                                       self.tflam: METHOD['lam']})
                if kl > 4 * METHOD['kl_target']:
                    break
                elif kl < METHOD[
                    'kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                    METHOD['lam'] /= 2
                elif kl > METHOD['kl_target'] * 1.5:
                    METHOD['lam'] *= 2
                METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)

        else:  # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op,
                           {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in
             range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in
         range(C_UPDATE_STEPS)]

    def update(self, grads_and_vars):
        self.actor_update_op = self.actor_opt.apply_gradients(grads_and_vars)
        self.critic_update_op = self.critic_opt.apply_gradients(grads_and_vars)
        self.sess.run([self.actor_update_op, self.critic_update_op])

    def get_grads_and_vars(self, s, a, r):
        adv = self.sess.run(self.advantage,
                            {self.tfs: s, self.tfdc_r: r})  # 得到advantage value
        self.actor_grads_and_vars_op = \
            self.actor_opt.compute_gradients(self.aloss,
                                             tf.trainable_variables(scope='pi'))
        self.critic_grads_and_vars_op = \
            self.critic_opt.compute_gradients(self.closs, tf.trainable_variables(
                scope='critic'))
        return self.sess.run([self.actor_grads_and_vars_op, self.critic_grads_and_vars_op],
                             {self.tfs: s, self.tfa: a, self.tfdc_r: r,
                              self.tfadv: adv})

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus,
                                    trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)  # 一个正态分布
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def set_weigths(self, params):
        self.a_variables.set_weights(params[0])
        self.c_variables.set_weights(params[1])

    def get_weights(self):
        return [self.a_variables.get_weights(), self.c_variables.get_weights()]


@ray.remote()
class RolloutWorker(object):
    def __init__(self):
        self.env = gym.make('Pendulum-v0').unwrapped
        with tf.Graph().as_default():
            self.model = PPO()
        self.opt = tf.train.AdamOptimizer(C_LR)
        )

    # 在主api函数里面传params
    def compute_grads_list_and_vars(self, params):
        self.model.set_weights(params)
        grads_and_vars = self._rollout()


    def _rollout(self):
        s = self.env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        actor_grads_list, critic_grads_list = [], []
        ep_r = 0
        for t in range(EP_LEN):  # in one episode
            # env.render()
            a = self.model.choose_action(s)
            s_, r, done, _ = self.env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8) / 8)  # normalize reward
            s = s_
            ep_r += r

            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                v_s_ = self.model.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(
                    discounted_r)[:, np.newaxis]
                actor_grads_and_vars, critic_grads_and_vars = self.model.get_gradients(bs, ba, br)
                actor_grads_list.append(actor_grads_and_vars[0])
                critic_grads_list.append(critic_grads_and_vars[0])
        return [actor_grads_list, critic_grads_list]


def main():
    env = gym.make('Pendulum-v0').unwrapped
    ppo = PPO()
    all_ep_r = []

    for ep in range(EP_MAX):
        s = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        ep_r = 0
        for t in range(EP_LEN):  # in one episode
            # env.render()
            a = ppo.choose_action(s)  # 根据一个正态分布，选择一个action
            s_, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
            s = s_
            ep_r += r

            # update ppo
            if (t + 1) % BATCH == 0 or t == EP_LEN - 1:
                v_s_ = ppo.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)  # v(s) = r + gamma * v(s+1)
                discounted_r.reverse()

                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(
                    discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                ppo.update(bs, ba, br)
        if ep == 0:
            all_ep_r.append(ep_r)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        print(
            'Ep: %i' % ep,
            "|Ep_r: %i" % ep_r,
            ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
        )

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()
