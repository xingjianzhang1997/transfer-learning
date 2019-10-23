import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
import matplotlib.pyplot as plt


INPUT_DATA = 'D:/python/deep-learning/MRI-2D/data_npy/data_ATR_L.npy'

# 保存训练模型
TRAIN_FILE = 'D:/python/deep-learning/MRI-2D/save_model/ '
# 谷歌训练好的模型
CKPT_FILE = 'inception_v3.ckpt'
LEARNING_RATE = 0.00005
STEPS = 100  # 480
BATCH = 32
N_CLASSES = 2


# 参数前缀
# 不需要加载的参数，这是最后的全连接层，在新的问题中需要重新训练这一层参数
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数名称
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'


# 获取所有需要从谷歌模型中的参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []

    # 枚举所有参数，判断是否需要
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return variables_to_restore


# 获取需要训练的变量
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variable_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variable_to_train.extend(variables)

    return variable_to_train


def main(self):
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    training_labels = processed_data[1]
    testing_images = processed_data[2]
    testing_labels = processed_data[3]
    n_training_example = len(training_images)

    print(len(training_images))
    print(len(training_labels))
    print(len(testing_images))
    print(len(testing_labels))

    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)

    trainable_variables = get_trainable_variables()

    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)

    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())  # 可以换成Adam试一下

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)  # 1表示的是按行比较返回最大值的索引
        A = tf.reduce_mean(tf.losses.get_total_loss())
        B = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        evaluation_step = [A, B]
    # 定义加载模型的参数
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)

    step_list = list(range(STEPS))  # [0,1,2,……,9]
    train_accuracy_list = []
    test_accuracy_list = []
    train_loss_list = []
    test_loss_list = []
    fig = plt.figure()  # 建立可视化图像框
    ax1 = fig.add_subplot(2, 3, 1)  # z子图总行数、列数，位置
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 4)
    ax4 = fig.add_subplot(2, 3, 5)
    ax5 = fig.add_subplot(2, 3, 3)
    ax1.set_title('cnn_train_accuracy', fontsize=10, y=1.02)
    ax2.set_title('cnn_test_accuracy', fontsize=10, y=1.02)
    ax3.set_title('cnn_train_loss', fontsize=10, y=1.02)
    ax4.set_title('cnn_test_loss', fontsize=10, y=1.02)
    ax5.set_title('cnn_ROC', fontsize=10, y=1.02)
    # ax1.set_xlabel('steps')
    # ax2.set_xlabel('steps')
    # ax3.set_xlabel('steps')
    # ax4.set_xlabel('steps')

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print('Loading tuned variables from %s' % CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):

            train_accuracy = sess.run(evaluation_step, feed_dict={
                images: training_images[start: end],
                labels: training_labels[start: end]
            })
            train_loss = train_accuracy[0]
            train_acc = train_accuracy[1]

            sess.run(train_step, feed_dict={
                images: training_images[start: end],
                labels: training_labels[start: end]
            })

            test_accuracy = sess.run(evaluation_step, feed_dict={
                images: testing_images,
                labels: testing_labels
            })
            test_loss = test_accuracy[0]
            test_acc = test_accuracy[1]

            train_accuracy_list.append(train_acc * 100.0)
            train_loss_list.append(train_loss)
            test_accuracy_list.append(test_acc * 100.0)
            test_loss_list.append(test_loss)

            if i % 10 == 0:
                print('Step %d: train loss = %.3f' % (i, train_loss))
                print('Step %d: train accruacy = %.1f%%' % (i, train_acc * 100.0))
                print('Step %d: Test loss = %.3f' % (i, test_loss))
                print('Step %d: Test accruacy = %.1f%%' % (i, test_acc * 100.0))

            if i == 99:
                # saver.save(sess, TRAIN_FILE, global_step=i)

                predict = sess.run(tf.nn.softmax(logits), feed_dict={
                    images: testing_images,
                    labels: testing_labels})
                AUC_pro = predict[:, 0]
                probability = predict[:, 0]*100  # 有18x8=144个
                TP = 0
                TN = 0
                FP = 0
                FN = 0
                FPR_list = []
                TPR_list = []
                for j in range(100):  # 101
                    for k in range(144):
                        if probability[k] <= j:  # 预测为1
                            if testing_labels[k] == 1:  # 真实为1
                                TP = TP + 1
                            elif testing_labels[k] == 0:  # 真实为0
                                FP = FP + 1
                        elif probability[k] > j:  # 预测为0
                            if testing_labels[k] == 1:  # 真实为1
                                FN = FN + 1
                            elif testing_labels[k] == 0:  # 真实为0
                                TN = TN + 1
                    FPR = FP/(FP+TN)
                    TPR = TP/(TP+FN)
                    FPR_list.append(FPR)
                    TPR_list.append(TPR)
                    TP = 0
                    TN = 0
                    FP = 0
                    FN = 0

                prediction_tensor = tf.convert_to_tensor(AUC_pro)
                label_tensor = tf.convert_to_tensor(testing_labels)
                auc_value, auc_op = tf.metrics.auc(label_tensor, prediction_tensor, num_thresholds=100)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                sess.run(auc_op)
                value = 1 - sess.run(auc_value)
                print("AUC:" + str(value))

            start = end
            if start == n_training_example:
                start = 0
            end = start + BATCH
            if end > n_training_example:
                end = n_training_example

        writer = tf.summary.FileWriter('D:/python/deep-learning/MRI-2D/BMP-cnn/to/log', tf.get_default_graph())
        writer.close()
        fig.tight_layout()
        ax1.plot(step_list, train_accuracy_list)
        ax2.plot(step_list, test_accuracy_list)
        ax3.plot(step_list, train_loss_list)
        ax4.plot(step_list, test_loss_list)
        # ROC曲线的横轴为假正例率(FPR), 纵轴为真正例率(TPR)
        ax5.plot(FPR_list, TPR_list)
        plt.show()


if __name__ == '__main__':
    tf.app.run()

