from tensorflow.compat import v1 as tf
import numpy as np
from matplotlib import pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# 神经网络之多输出归纳：每个输出之间相互独立
class Config:
    def __init__(self):
        self.lr = 0.01
        self.epoches = 2000
        self.save_path = "models/p33/sin_cos"
        self.train_size = 200
        self.predict_size = 200
        self.hidden_units = 200
        self.train_range = np.pi
        self.predict_range = np.pi


class Simple:
    def __init__(self,low,high,example_size):
        self.xs = np.random.uniform(low=low, high=high, size=example_size) # 一维张量
        self.xs = sorted(self.xs)
        self.ys = np.sin(self.xs) , np.cos(self.xs) , np.square(self.xs) # 二维张量 [2,-1]

    @property
    def num_example(self):
        return len(self.xs)

class Tensors:
    def __init__(self,config : Config):
        self.xs = tf.placeholder(dtype=tf.float32, shape=[None], name="x") # 输入的x值为一维张量 , 如[1，2，3，4]
        x = tf.reshape(self.xs, [-1,1]) # 将x标量转化为[n,1]的二维张量 ， n为输入时的样本量
        x = tf.layers.dense(x, config.hidden_units,activation=tf.nn.relu) # 全连接层
        self.y_predict = tf.layers.dense(x, 3) # 输出层， 输出为[n,2]的二维张量
        y_predict = tf.transpose(self.y_predict)  # 转置 [2,n]

        self.ys = tf.placeholder(dtype=tf.float32,shape=[3,None], name="y") # 真实值，
        self.lr = tf.placeholder(dtype=tf.float32,name="lr")
        self.loss = tf.reduce_mean(tf.square(self.ys - y_predict))
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_opt = opt.minimize(self.loss)


class SinApp:
    def __init__(self,config : Config):
        self.config = config
        self.ts = Tensors(config)
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())
        try:
            self.saver.restore(self.session,self.config.save_path)
        except:
            self.session.run(tf.global_variables_initializer())

    def train(self):
        config = self.config
        simple = Simple(-config.train_range, config.train_range, config.train_size)
        for epoch in range(config.epoches):
            _,loss = self.session.run([self.ts.train_opt, self.ts.loss],
                             {self.ts.xs : simple.xs,
                              self.ts.ys : simple.ys,
                              self.ts.lr : config.lr})
            print(f"epoch = {epoch} , loss = {loss}")
        self.save()
        return simple.xs, simple.ys

    def predict(self):
        config = self.config
        simple = Simple(-config.predict_range,config.predict_range,config.predict_size)
        xs = simple.xs
        y_predict = self.session.run(self.ts.y_predict, {self.ts.xs : xs})
        return xs,y_predict

    def save(self):
        self.saver.save(self.session,self.config.save_path)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__':
    tf.disable_eager_execution()

    app = SinApp(Config())
    with app:
        train_xs,train_ys = app.train()
        predict_xs,predict_ys = app.predict()
    train_ys = np.transpose(train_ys)
    plt.plot(train_xs,train_ys)
    plt.plot(predict_xs,predict_ys)
    plt.show()













