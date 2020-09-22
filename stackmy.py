import tensorflow as tf
import numpy as np

learn_rate = 0.1

W_init = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
x_init = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]], dtype=np.float32)

with tf.variable_scope('loss'):
    X = tf.placeholder(tf.float32, name="x")
    W = tf.Variable(W_init, dtype=np.float32, name="w")
    y = tf.matmul(X, W, name="y")
    loss = tf.reduce_mean(y, name="loss")

grad = tf.placeholder(tf.float32, name='grad')
opt = tf.train.AdamOptimizer(learn_rate)

t_vars = tf.trainable_variables()
# 初始化accum variable list
accum_tvars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
               for tv in t_vars]
# accum tensor置零op
zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_tvars]

# 计算单次梯度的op
batch_grads_vars = opt.compute_gradients(loss, t_vars)
# 将单个梯度assign add到accum里
accum_ops = [accum_tvars[i].assign_add(batch_grad_var[0]) for i, batch_grad_var in
             enumerate(batch_grads_vars)]

# apply accums gradients
train_step = opt.apply_gradients(
    [(accum_tvars[i], batch_grad_var[1]) for i, batch_grad_var in
     enumerate(batch_grads_vars)])
# train_step = opt.apply_gradients(zip(accum_tvars, zip(*batch_grads_vars)[1])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(10):
    # initialize the accumulated gards
    sess.run(zero_ops)

    # number of batches for gradient accumulation
    n_batches = 3
    for i in range(n_batches):
        print(i)
        # sess.run(accum_ops, feed_dict={X: x_init[None, i]})
        grad_and_var = sess.run(batch_grads_vars, feed_dict={X: x_init[None, i]})
        print('grad', grad_and_var)
        # print('var', grad_and_var[1])

    sess.run(train_step)

print("Weights after 1 gradient")
print(sess.run(W))