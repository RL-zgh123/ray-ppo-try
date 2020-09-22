import tensorflow as tf
import numpy as np

a = tf.Variable(1.0, name = 'a')
b = tf.Variable(2.0, name = 'b')
ema = tf.train.ExponentialMovingAverage(decay=0.9999)
d = ema.apply([a, b])
c = tf.reduce_mean([a, b])
print(a, b)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run((a+b)/2))
print(sess.run(d))
print(sess.run(c))