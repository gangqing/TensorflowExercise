tensorflow2.0时，还需要调用tf.disable_eager_execution()，这函数只能在创建任何图、操作或张量之前调用。它可以在程序开始时用于从TensorFlow 1.x到2.x的复杂迁移项目。

### 题目29:
使用tensorflow定义变量，求开根.
> 初识session(图),variable(变量)，placeholder(占位符)，tensorflow自动求导。

### 题目30:
使用tensorflow自定义神经网络，批量求开根，使用matplotlib绘制结果图，求导交给tensorflow。

### 题目31:
使用神经网络训练并预测MINST数据集。

### 题目32:
使用神经网络训练并预测sin函数数据集。

### 题目33:
使用神经网络进行多输出练习。

### 题目34:
使用卷积神经网络训练并预测MINST数据集。

### 题目35:
在题目34的基础上添加loss和precise的summary，使用tensorboard查看summary.

>tensorboard命令：tensorboard --logdir logs/p35 --port 9876
>对应的网址：http://localhost:9876

### 题目36:
抽取config类，使用argparse，让config能从终端中更改变量值。如：
>python3 p36_config.py --lr 0.1

### 题目37：
在题目35上修改，指定使用0号GPU创建Tensors
>tf.device("/gpu:0")

### 题目38
使用argparse创建变量

### 题目39
整合p36和p37，抽取不变部分作为框架


### 题目40
使用题目39的框架，实现VAE模型

### 题目41
编写VAE模型，生成随机数字的图像，并将多张图像合成一张图像输出

### 题目42
编写CVAE模型，生成指定数字的图像，并将多张图像合成一张图像输出





