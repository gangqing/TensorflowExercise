from tensorflow_core.examples.tutorials.mnist.input_data import read_data_sets
from tensorflow.compat import v1 as tf

tf.disable_eager_execution()
# 神经网络之MINST图片分类
class Tensors:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None,784], name="x") # 输入[-1 * 784]， -1为未知的样本数， 784为特征数
        x = tf.layers.dense(self.x,units=2000,activation=tf.nn.relu) # 全连接层，输出为2000个特征[-1，2000] ， 使用relu激活函数
        x = tf.layers.dense(x,units=10) # 全连接层，输出为[-1,10], 10是因为有10个类别0-9
        y_predict = tf.nn.softmax(x) # 使用sortmax将输出转为概率
        y_predict = tf.maximum(y_predict,1e-6) # 当概率小于1e-6时，使用1e-6代替原值
        self.y_predict = tf.argmax(y_predict,axis=1,output_type=tf.int32)

        self.y = tf.placeholder(dtype=tf.int32,shape=[None],name="y") # 标签值，0-9
        y = tf.one_hot(self.y,10) # 将标签值使用独热编码转换成[-1,10]的形式

        self.loss = - tf.reduce_mean(tf.reduce_sum(y * tf.log(y_predict) , axis=1))
        self.lr = tf.placeholder(dtype=tf.float32, name="lr")

        opt = tf.train.AdamOptimizer(self.lr)
        self.train_opt = opt.minimize(self.loss)

        self.precise = tf.reduce_mean(tf.cast(tf.equal(self.y , self.y_predict) , dtype=tf.float32))

class MNISTApp:
    def __init__(self,save_path):
        self.save_path = save_path
        self.tensors = Tensors()
        self.session = tf.Session()
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.session,save_path)
            print("restore model success")
        except:
            self.session.run(tf.global_variables_initializer())
            print("restore model fail")

    def train(self,ds,lr=0.01,batch_size=200,epoches=20):
        for epoch in range(epoches):
            batches = ds.train.num_examples // batch_size
            for batch in range(batches):
                xs,ys = ds.train.next_batch(batch_size)
                _,loss = self.session.run([self.tensors.train_opt,self.tensors.loss],
                                          {self.tensors.x : xs , self.tensors.y : ys , self.tensors.lr : lr})
                print(f"epoches={epoch} , batch = {batch} , loss = {loss}")
            self.valid(ds)
        self.save()

    def predict(self,ds,batch_size=200):
        xs,ys = ds.test.next_batch(batch_size)
        ys_predict = self.session.run(self.tensors.y_predict,{self.tensors.x : xs})
        print("predict")
        return ys,ys_predict

    def valid(self,ds,batch_size=200):
        xs, ys = ds.validation.next_batch(batch_size)
        precise = self.session.run(self.tensors.precise, {self.tensors.x: xs, self.tensors.y : ys})
        return precise

    def save(self):
        self.saver.save(self.session,self.save_path)

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    path = "MNIST_data"
    ds = read_data_sets(path)
    print(ds.train.num_examples) # 打印 训练集的数量
    print(ds.test.num_examples) # 打印 测试集的数量
    print(ds.validation.num_examples) # 打印 验证集的数量

    save_path = "models/p31/mnist"
    app = MNISTApp(save_path)
    with app:
        app.train(ds)
        app.predict(ds)
