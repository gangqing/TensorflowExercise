import argparse
from tensorflow.compat import v1 as tf
from tensorflow_core.examples.tutorials.mnist.input_data import read_data_sets
import os


def get_gpus():
    value = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    value = value.split(",")
    return value


def make_dirs(path: str):
    pos = path.rfind(os.sep)
    if pos < 0:
        raise Exception('Can not find the directory from the path', path)
    path = path[:pos]
    os.makedirs(path, exist_ok=True)


class Config:
    def __init__(self):
        tf.disable_eager_execution()
        self.save_path = "models/{name}/{name}".format(name=self.get_name())
        self.simple_path = "MNIST_data"
        self.logdir = "logs/{name}".format(name=self.get_name())
        self.lr = 0.001
        self.epoches = 2
        self.batch_size = 200
        self.new_model = False
        self.gpus = get_gpus()

    def get_name(self):
        raise Exception("get_name() is not re-defined !")

    def __repr__(self):
        attrs = self.get_attrs()  # 字典
        result = ["{attr} = {value}".format(attr=attr, value=attrs[attr]) for attr in attrs]

        return ", ".join(result)

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith("_"):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in [int, float, bool, str]:
                result[attr] = value
        return result

    def from_cmd(self):
        parse = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = getattr(self, attr)
            if type(value) == bool:
                parse.add_argument(f"--{attr}", default=value, help=f"default to {value}",
                                   action=f"store_{not value}".lower())
            else:
                parse.add_argument(f"--{attr}", type=type(value), default=value, help=f"default to {value}")
        arg = parse.parse_args()
        for attr in attrs:
            if hasattr(arg, attr):
                setattr(self, attr, getattr(arg, attr))

    def get_tensors(self):
        raise Exception("get_tensors() is not re-defined !")

    def get_app(self):
        return App(self)


class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()  # 图
        with graph.as_default():
            self.ts = config.get_tensors()
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(graph=graph, config=cfg)
            self.saver = tf.train.Saver()
            if config.new_model:
                self.session.run(tf.global_variables_initializer())
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                except:
                    self.session.run(tf.global_variables_initializer())

    def train(self, ds_train, ds_validation):
        config = self.config
        session = self.session
        ts = self.ts
        writer = tf.summary.FileWriter(logdir=config.logdir, graph=self.session.graph)
        batches = ds_train.num_examples // config.batch_size
        for epoch in range(config.epoches):
            for batch in range(batches):
                _, loss = session.run([ts.train_op, ts.loss_summary], self.get_feed_dict(ds_train))
                writer.add_summary(loss, global_step=epoch * batches + batch)
                print(f"epoch = {epoch} , batch = {batch} , loss = {loss}")
            # precise = session.run(ts.precise_summary, self.get_feed_dict(ds_validation))
            # writer.add_summary(precise, global_step=epoch)
        self.save()

    def get_feed_dict(self, ds):
        values = ds.next_batch(self.config.batch_size)  # xs,ys
        return {tensor: value for tensor, value in zip(self.ts.inputs, values)}

    def save(self):
        self.saver.save(self.session, save_path=self.config.save_path)

    def test(self, ds_test):
        pass

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    cfg = Config()
    cfg.from_cmd()
    print(cfg)

    ds = read_data_sets(cfg.simple_path)
    app = cfg.get_app()
    with app:
        app.train(ds_train=ds.train, ds_validation=ds.validation)
        app.test(ds_test=ds.test)
