from tensorflow.compat import v1 as tf
from matplotlib import pyplot as plt
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
tf.disable_eager_execution()

class Ds:
    def __init__(self):
        self.train_xs = np.array([x/100 for x in range(200)])
        self.train_ys = np.sqrt(self.train_xs)

        self.pre_xs = sorted(np.random.uniform(0, 2, [400]))

class SqrtApp:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None], name="x") # [-1]
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name="y") # [-1]
        self.lr = tf.placeholder(dtype=tf.float32, shape=None, name="lr") # 0.01

        x = tf.reshape(self.x, [-1,1]) # [-1,1]
        w1 = tf.get_variable(dtype=tf.float32, shape=[1, 200], name="w1") # [1, 200]
        b1 = tf.get_variable(dtype=tf.float32, shape=[200], name="b1") # [200]

        x = tf.matmul(x, w1) + b1 # [-1, 200]
        x = tf.nn.relu(x)
        w2 = tf.get_variable(dtype=tf.float32, shape=[200, 1], name="w2") # [1, 200]
        b2 = tf.get_variable(dtype=tf.float32, shape=[1], name="b2")  # [1]

        self.pre_y = tf.matmul(x, w2) + b2 # [-1, 1]
        y = tf.reshape(self.y, [-1,1])

        loss = tf.reduce_mean(tf.square(self.pre_y - y))
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self, xs, ys, lr=0.01, epoches=2000):
        for _ in range(epoches):
            self.session.run(self.train_op, {self.x : xs, self.y : ys, self.lr : lr})


    def predict(self, xs):
        return self.session.run(self.pre_y,{self.x : xs})

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__':
    ds = Ds()
    app = SqrtApp()

    with app:
        xs = ds.train_xs
        ys = ds.train_ys
        app.train(xs, ys)

        pre_xs = ds.pre_xs
        pre_ys = app.predict(pre_xs)

    plt.plot(xs,ys)
    plt.plot(pre_xs,pre_ys)
    plt.show()



