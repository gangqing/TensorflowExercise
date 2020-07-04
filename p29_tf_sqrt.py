from tensorflow.compat import v1 as tf

# 初识tensorflow , 使用tensorflow解决sqrt求解
tf.disable_eager_execution()

result = tf.get_variable(name="result", shape=(), dtype=tf.float32)
n = tf.placeholder(name="n", shape=(), dtype=tf.float32)
lr = tf.placeholder(name="lr", shape=(), dtype=tf.float32)

loss = (result ** 2 - n) ** 2
opt = tf.train.GradientDescentOptimizer(lr)
train_opt = opt.minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

def sqrt(number, lr_v=0.01, epoches=2000):
    for _ in range(epoches):
        session.run(train_opt, {n: number, lr: lr_v})
    return session.run(tf.abs(result))


if __name__ == '__main__':
    for i in range(1, 10 + 1):
        print(f"sqrt({i} = {sqrt(i)})")
    session.close()
