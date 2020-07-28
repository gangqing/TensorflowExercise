
from tensorflow.compat import v1 as tf
from tensorflow_core.examples.tutorials.mnist.input_data import read_data_sets

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 使用二维卷积神经网络进进MINST图片分类

class Config():
    def __init__(self):
        self.data_path = "MNIST_data"
        self.save_path = 'models/p37/GPU'
        self.logdir = "logs/p37"
        self.batch_size = 200
        self.lr = 0.01
        self.epoches = 1
        self.is_new_model = True

class Tensors():
    def __init__(self):
        with tf.device("/gpu:0"):
            # 初始化数据
            self.xs = tf.placeholder(dtype=tf.float32, shape=[None,784], name="x") # [n,784]
            self.ys = tf.placeholder(dtype=tf.int32, shape=[None], name="y") #[n]
            self.lr = tf.placeholder(dtype=tf.float32, shape=None, name="lr") # 0.01
            xs = tf.reshape(self.xs,[-1,28,28,1])
            # 隐层和输出层
            xs = tf.layers.conv2d(xs,16,3,strides=(1, 1),padding='same',activation=tf.nn.relu) # [-1,28,28,16]
            xs = tf.layers.conv2d(xs,32,3,2,padding="same",activation=tf.nn.relu) # [-1,14,14,32]
            xs = tf.layers.conv2d(xs,64,3,2,padding="same",activation=tf.nn.relu) # [-1,7,7,64]
            xs = tf.layers.flatten(xs) # [-1, 7*7*64]
            ys_predict = tf.layers.dense(xs,10) # [-1, 10]
            # 输出结果
            self.ys_predict = tf.argmax(ys_predict, axis=1,output_type=tf.int32) # 标签
            precise = tf.cast(tf.equal(self.ys,self.ys_predict),dtype=tf.float32)
            self.precise = tf.reduce_mean(precise) # 精度
            self.precise_summary = tf.summary.scalar(name="precise_summary", tensor=self.precise)
            # 损失函数
            ys = tf.one_hot(self.ys, 10)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = ys, logits = ys_predict) # [-1,1]
            self.loss = tf.reduce_mean(loss)
            self.loss_summary = tf.summary.scalar(name="loss_summary",tensor=self.loss)
            # 优化器
            opt = tf.train.AdamOptimizer(self.lr)
            self.train_opt = opt.minimize(self.loss)



class App():
    def __init__(self,config : Config):
        cfp = tf.ConfigProto()  # tf.ConfigProto一般用在创建session的时候用来对session进行参数配置。
        cfp.allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备

        self.config = config
        self.session = tf.Session(config = cfp)
        self.ts = Tensors()
        self.saver = tf.train.Saver()

        if config.is_new_model:
            self.session.run(tf.global_variables_initializer())
        else:
            try:
                self.saver.restore(self.session,config.save_path)
            except:
                self.session.run(tf.global_variables_initializer())

    def train(self,da):
        writer = tf.summary.FileWriter(logdir=self.config.logdir,graph=self.session.graph)
        batches = da.train.num_examples // self.config.batch_size
        for epoch in range(self.config.epoches):
            for batch in range(batches):
                xs,ys = da.train.next_batch(self.config.batch_size)
                _,loss_summary = self.session.run([self.ts.train_opt,self.ts.loss_summary],{self.ts.xs : xs , self.ts.ys : ys, self.ts.lr : self.config.lr})
                writer.add_summary(summary=loss_summary, global_step=epoch * batches + batch)

            vxs,vys = da.validation.next_batch(self.config.batch_size)
            precise_summary = self.session.run(self.ts.precise_summary,{self.ts.xs : vxs ,self.ts.ys : vys})
            writer.add_summary(precise_summary,epoch * batches)
        self.save()

    def save(self):
        self.saver.save(self.session,self.config.save_path)

    def predict(self,da):
        precise_totle = 0
        batches = da.test.num_examples // self.config.batch_size
        for batch in range(batches):
            xs,ys = da.test.next_batch(self.config.batch_size)
            precise = self.session.run(self.ts.precise,{self.ts.xs : xs , self.ts.ys : ys})
            precise_totle += precise
        print(f"precise = {precise_totle / batches}")

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    tf.disable_eager_execution()
    tf.reset_default_graph()

    config = Config()
    app = App(config)
    da = read_data_sets(config.data_path)
    with app:
        app.train(da)
        # app.predict(da)

    app = App(config)
    with app:
        app.train(da)
        # app.predict(da)
