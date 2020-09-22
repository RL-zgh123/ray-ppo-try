import tensorflow as tf
import numpy as np

learn_rate = 0.1

W_init = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
b_init = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
x_init = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]], dtype=np.float32)

with tf.variable_scope('layer1'):
    X = tf.placeholder(tf.float32, name="x")
    W = tf.Variable(W_init, dtype=np.float32, name="w")
    b = tf.Variable(b_init, dtype=np.float32, name="b")
    y = tf.add(tf.matmul(X, W, name="y_mul"), b, name="y_add")
    loss = tf.reduce_mean(y, name="loss")

opt = tf.train.AdamOptimizer(learn_rate)
avg_grads_and_vars = []
t_vars = tf.trainable_variables()
grads_and_vars = opt.compute_gradients(loss, t_vars)  # (array, array)
grad_placeholder = []

# 出图之后的梯度值如图操作
# print(grads_and_vars)
dic = {}
ph_dic = {}
for grad, var in grads_and_vars: # [(grad, var)]
    dic[var] = grad
    ph_dic[var] = tf.placeholder(grad.dtype, grad.shape)
    grad_placeholder.append(ph_dic[var])
    avg_grads_and_vars.append((ph_dic[var], var))

train0_op = opt.apply_gradients(avg_grads_and_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

grad_sum = {}
n_batches = 3

# sum and average gradients
for i in range(n_batches):
    # 计算梯度值在worker里完成
    for var in dic:
        grad = sess.run(dic[var], {X: x_init[None, i]})
        print(var, grad)
        if var not in grad_sum:
            grad_sum[var] = grad
        else:
            grad_sum[var][0] += grad[0]
for var in dic:
    grad_sum[var][0] /= n_batches
    print(var, grad_sum[var])

sess.run(train0_op, {ph_dic[var]: grad_sum[var] for var in grad_sum})

print("Weights after 1 gradient")
print(sess.run(W))
