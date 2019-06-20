import os
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import alexnet
from keras.utils import to_categorical
import time
import math
train_path = '.\\mstar\\train'
test_path = '.\\mstar\\val'

def countFile(dir):
    tmp = 0
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, item)):
            tmp += 1
        else:
            tmp += countFile(os.path.join(dir, item))
    return tmp

class load_mstar():

    def __init__(self,super_path,batch_size):
        self.super_path = super_path
        self.batch_size = batch_size
        self.gen_batches = self.batch_generator()
        self.num_files = countFile(super_path)
        self.num_batches = math.ceil(self.num_files/self.batch_size)

    def get_files_labels(self):
        file_list = []
        label_list = []
        for path_i in os.listdir(self.super_path):
            path_ = path_i
            path_i = os.path.join(self.super_path,path_i)
            for path_j in os.listdir(path_i):
                # path_j = os.path.join(path_i,path_j)
                file_list.append(os.path.join(path_,path_j))
                label_list.append(int(path_))
        return file_list, label_list


    def batch_generator(self):
        file_list, label_list = self.get_files_labels()
        random.seed(10)
        random.shuffle(file_list)
        random.seed(10)
        random.shuffle(label_list)

        num_batchs = len(file_list)/self.batch_size
        i = 0
        while True:
            images_batch = []
            ii = int(np.mod(i,num_batchs))
            files_batch = file_list[0+ii*self.batch_size:(ii+1)*self.batch_size]
            labels_batch = label_list[0+ii*self.batch_size:(ii+1)*self.batch_size]
            i += 1
            for j in files_batch:
                image = cv2.imread(os.path.join(self.super_path,j))
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = image/255
                images_batch.append(image)

            try:
                yield np.array(images_batch), to_categorical(np.array(labels_batch),num_classes=10)
            except Exception:
                print('End Queue!')

    def get_batch(self):

        batch_x, batch_y = next(self.gen_batches)

        return batch_x, batch_y


    def get_num_batches(self):
        file_list, label_list = self.get_files_labels()
        random.seed(10)
        random.shuffle(file_list)
        random.seed(10)
        random.shuffle(label_list)

        num_batchs = len(file_list) / self.batch_size

# def main():

class nets():
    def __init__(self,sess):
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.w = 128
        self.h = 128
        self.c = 3
        self.test_size = 32
        self.class_nums = 10
        self.train_data = load_mstar(super_path=train_path,batch_size=self.batch_size)
        self.test_data = load_mstar(super_path=test_path,batch_size=self.batch_size)
        self.sess = sess
        self.epoch = 100
        self.logs_dir = './logs'
        self.keep_rate = 0.5
    def networks(self,x,scope='cnn',reuse=False,is_training=True):
        with tf.variable_scope(scope,reuse=reuse):
            net = tf.layers.conv2d(x,32,3,2,'SAME',activation=tf.nn.relu)
            net = tf.layers.dropout(net,rate=self.keep_rate,training=is_training)
            net = tf.layers.conv2d(net, 32, 3, 2, 'SAME', activation=tf.nn.relu)
            net = tf.layers.dropout(net,rate=self.keep_rate,training=is_training)
            net = tf.layers.flatten(net)
            net = tf.layers.dense(net,512,activation=tf.nn.relu)
            net = tf.layers.dropout(net,rate=self.keep_rate,training=is_training)
            logits = tf.layers.dense(net,self.class_nums)
        return logits
    def build_model(self):

        # 训练数据和标签的placeholder
        self.train_x = tf.placeholder(tf.float32,[None,self.w,self.h,self.c], name='train_x')
        self.train_y = tf.placeholder(tf.float32,[None,self.class_nums], name='train_y')
        # 测试数据和标签的placeholder
        self.test_x = tf.placeholder(tf.float32,[None,self.w,self.h,self.c], name='test_x')
        self.test_y = tf.placeholder(tf.float32,[None, self.class_nums])

        slim = tf.contrib.slim

        self.train_logits = self.networks(self.train_x)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.train_logits, labels=self.train_y)
        self.loss = tf.reduce_mean(self.loss)
        self.optm = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # 测试计算准确率
        self.test_logits = self.networks(self.test_x,reuse=True,is_training=False)
        correct_prediction = tf.equal(tf.argmax(self.test_logits, 1), tf.argmax(self.test_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # summary
        tf.summary.scalar('train_loss', self.loss)
        self.merge_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        for i in range(self.epoch*self.train_data.num_batches):
            batch_x, batch_y = self.train_data.get_batch()
            train_dict = {self.train_x:batch_x,
                          self.train_y:batch_y}
            train_logits,train_loss, _ = self.sess.run([self.train_logits,self.loss, self.optm],feed_dict=train_dict)

            train_summary = self.sess.run(self.merge_summary, feed_dict=train_dict)
            self.train_writer.add_summary(train_summary,i)
            print('train_loss %.6f' % train_loss)

            # 计算测试准确率
            if np.mod(i, 300) == 0.0:
                start_time = time.time()
                acc_test = 0
                for j in range(self.test_data.num_batches):
                    batch_x_test, batch_y_test = self.test_data.get_batch()
                    test_dict = {self.test_x:batch_x_test,
                                 self.test_y:batch_y_test}
                    acc_batch_test = self.sess.run(self.accuracy,feed_dict=test_dict)
                    acc_test += acc_batch_test

                acc_test /= self.test_data.num_batches

                print('测试准确率 %.6f' % acc_test)
                print('测试耗时 %.2f' % (time.time()-start_time))
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        slover = nets(sess)
        slover.build_model()

        slover.train()

if __name__  == '__main__':
    main()