# coding:utf-8

import os

import tensorflow as tf

import BRCA_inference
import BRCA_batch
import BRCA_data

# 配置神经网络参数
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

# 数据读取路径
Data_Read_PATH = "/home/sunysh/12Cancer/Total-2/part1/train.Matrix"

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "/home/sunysh/12Cancer/Total-2/part1/Model"
MODEL_NAME = "model.ckpt"

def train():
    # 训练数据读取
    L = BRCA_data.readCase(Data_Read_PATH)

    #定义输入层
    x = tf.placeholder(tf.float32, [None, BRCA_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, BRCA_inference.OUTPUT_NODE], name='y_input')

    # 返回regularizer函数，L2正则化项的值
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 使用BRCA_inference.py中定义的前向传播过程
    y = BRCA_inference.inference(x, regularizer)
    # 定义step为0
    global_step = tf.Variable(0, trainable=False)

    # 滑动平均,由衰减率和步数确定
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 可训练参数的集合
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 交叉熵损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 总损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 学习率(衰减)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, TRAINING_STEPS, LEARNING_RATE_DECAY)
    # 定义了反向传播的优化方法，之后通过sess.run(train_step)就可以对所有GraphKeys.TRAINABLE_VARIABLES集合中的变量进行优化，似的当前batch下损失函数更小
    # 实现梯度下降算法的优化器
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 更新参数
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    #TensorBoard
    writer = tf.summary.FileWriter("/home/sunysh/12Cancer/Total-2/part1/Log", tf.get_default_graph())
    writer.close()

    # 初始会话，并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            L = BRCA_data.readCase(Data_Read_PATH)
            xs, ys = BRCA_batch.all(L)
            op, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            print(global_step.eval(), 'loss:', loss_value)
	    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        #saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
