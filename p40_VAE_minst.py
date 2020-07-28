import p39_framework as myf
from p39_framework import tf
from tensorflow_core.examples.tutorials.mnist.input_data import read_data_sets


class MyConfig(myf.Config):
    def __init__(self):
        super().__init__()
        self.vec_size = 4
        self.momentum = 0.99

    def get_name(self):
        return "p40"

    def get_tensors(self):
        return MyTensors(self)


class MyTensors:
    def __init__(self, config: MyConfig):
        self.config = config
        with tf.device("/gpu:0"):
            self.xs = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="x")  # [-1,784]
            self.ys = tf.placeholder(dtype=tf.int32, shape=[None], name="y")  # [-1]
            self.lr = tf.placeholder(dtype=tf.float32, shape=None, name="lr")  #
            self.inputs = [self.xs, self.ys, self.lr]

            x = tf.reshape(self.xs, [-1, 28, 28, 1])
            vec = self.encode(x, config.vec_size)
            y = self.decode(vec)
            # 计算vec的平均值
            self.process_normal(vec)

            loss = tf.reduce_mean(tf.square(y - x))
            opt = tf.train.AdamOptimizer(self.lr)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_opt = opt.minimize(loss)
            self.loss_summary = tf.summary.scalar(name="loss", tensor=tf.sqrt(loss))
            self.precise_summary = None

    def process_normal(self, vec):
        """"
        :param vec:  [-1, 4]
        """""
        mean = tf.reduce_mean(vec, axis=0)  # 当前平均值
        vec_size = vec.shape[1]
        self.final_mean = tf.get_variable(name="mean", shape=[vec_size], dtype=tf.float32, trainable=False)  # 目标平均值
        momentum = self.config.momentum
        assign = tf.assign(self.final_mean, self.final_mean * momentum + mean * (1 - momentum))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, assign)

    def encode(self, x, vec_size):
        """"
        :param x: [-1,28,28,1]
        :param vec_size: 4
        :return: [-1,1,1,vec_size]
        """""
        x = tf.layers.conv2d(x, 16, 3, strides=1, padding="same", activation=tf.nn.relu,
                             name="en_conv2_0")  # [-1, 28, 28, 16]
        x = tf.layers.conv2d(x, 32, 3, strides=1, padding="same", activation=tf.nn.relu,
                             name="en_conv2_1")  # [-1, 28, 28, 32]
        x = tf.layers.max_pooling2d(x, 2, 2, padding='valid', name="en_max_pool2_1")  # [-1, 14, 14, 32]
        x = tf.layers.conv2d(x, 64, 3, strides=1, padding="same", activation=tf.nn.relu,
                             name="en_conv2_2")  # [-1, 14, 14, 64]
        x = tf.layers.max_pooling2d(x, 2, 2, padding='valid', name="en_max_pool2_2")  # [-1, 7, 7, 64]
        x = tf.layers.flatten(x, name="en_flatten")  # [-1, 7 * 7 * 64]
        vec = tf.layers.dense(x, units=vec_size, name="en_dense")  # [-1, vec_size]
        return vec

    def decode(self, vec):
        """"
        :param vec: [-1, vec_size]
        :return: [-1, 28, 28, 1]
        """""
        y = tf.layers.dense(vec, units=7 * 7 * 64, name="de_dense")  # [-1, 7 * 7 * 64]
        y = tf.reshape(y, [-1, 7, 7, 64])  # [-1, 7, 7, 64]
        y = tf.layers.conv2d_transpose(y, 32, 3, strides=2, padding="same", activation=tf.nn.relu,
                                       name="de_conv2_0")  # [-1, 14, 14, 32]
        y = tf.layers.conv2d_transpose(y, 16, 3, strides=2, padding="same", activation=tf.nn.relu,
                                       name="de_conv2_1")  # [-1, 28, 28, 16]
        y = tf.layers.conv2d_transpose(y, 1, 3, strides=1, padding="same", activation=tf.nn.relu,
                                       name="de_conv2_2")  # [-1, 28, 28, 1]
        return y


class MyDS:
    def __init__(self, ds, config):
        self.ds = ds
        self.cfg = config

    def next_batch(self):
        xs, ys = self.ds.next_batch(self.cfg.batch_size)
        return xs, ys, self.cfg.lr

    @property
    def num_examples(self):
        return self.ds.num_examples


if __name__ == '__main__':
    tf.disable_eager_execution()
    tf.reset_default_graph()

    config = MyConfig()
    ds = read_data_sets(config.simple_path)
    app = myf.App(config)
    with app:
        app.train(ds_train=MyDS(ds.train, config), ds_validation=MyDS(ds.validation, config))
        mean = app.session.run(app.ts.final_mean)
        print(mean)
