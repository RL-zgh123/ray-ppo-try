import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import ray
from ray.experimental.tf_utils import TensorFlowVariables

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
METHOD = dict(name='clip', epsilon=0.2)

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






def main():
    ray.init()


if __name__ == "__main__":
    main()