import glob
import os
import numpy as np
import tensorflow as tf
import random


from tensorflow.python.platform import gfile


# fornix_L ILF_L Cingulum_hippocampus_L ATR_L
INPUT_DATA = 'D:/python/deep-learning/MRI-2D/BMP-cnn/BMP-data/fornix_L'
# 以numpy的格式保存图片数据
OUTPUT_FILE = 'D:/python/deep-learning/MRI-2D/data_npy/data.npy'


def create_image_list(sess):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)][1:]  # sub_dirs是所有子文件名

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    current_label = 0
    chance = 0  # 用来计数，表示是第几张图片
    list_num_0 = []  # 8个人
    list_num_1 = []  # 10个人
    list_test_num_0 = []  # 8 * 8张图
    list_test_num_1 = []  # 10 * 8张图

    # 构建分训练集和测试集的列表 label=0
    for i in range(42):  # 从0开始到41
        number = i * 8 + 1  # 因为一个人有8张图片
        list_num_0.append(number)  # 最大的number应该等于329，一共有42个

    list_test = random.sample(list_num_0, 8)  # 任选出8个人为测试集

    for i in list_test:
        list_test_num_0.append(i)
        list_test_num_0.append(i + 1)
        list_test_num_0.append(i + 2)
        list_test_num_0.append(i + 3)
        list_test_num_0.append(i + 4)
        list_test_num_0.append(i + 5)
        list_test_num_0.append(i + 6)
        list_test_num_0.append(i + 7)

    # 构建分训练集和测试集的列表 label=1
    for i in range(54):  # 从0开始到53
        number = i * 8 + 337  # 因为一个人有8张图片,337是第一个的图片序号
        list_num_1.append(number)  # 最大的number应该等于761，一共有54个

    list_test = random.sample(list_num_1, 10)  # 任选出10个人为测试集

    for i in list_test:
        list_test_num_1.append(i)
        list_test_num_1.append(i + 1)
        list_test_num_1.append(i + 2)
        list_test_num_1.append(i + 3)
        list_test_num_1.append(i + 4)
        list_test_num_1.append(i + 5)
        list_test_num_1.append(i + 6)
        list_test_num_1.append(i + 7)

    print("***")
    print(len(list_test_num_0))
    print(len(list_test_num_1))

    for sub_dir in sub_dirs:
        extensions = ['bmp']
        file_list = []
        dir_name = os.path.basename(sub_dir)  # os.path.basename(),返回path最后的文件名
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        print(len(file_list))  # 把一个子文件里的所有图片加入到file_list里面。

        for file_name in file_list:
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_bmp(image_raw_data, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299],  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # raw_data=[46,810]=37260
            # 问题出在inception-V3 只认299*299*3的彩色图片
            image_value = sess.run(image)
            chance = chance + 1  # chance是处理图片的数量

            if chance < 337:  # label = 0
                if chance in list_test_num_0:  # 开始查表，作为测试集
                    testing_images.append(image_value)
                    testing_labels.append(current_label)
                else:
                    training_images.append(image_value)
                    training_labels.append(current_label)
            elif 336 < chance < 769:  # label = 1
                if chance in list_test_num_1:  # 开始查表，作为测试集
                    testing_images.append(image_value)
                    testing_labels.append(current_label)
                else:
                    training_images.append(image_value)
                    training_labels.append(current_label)

        current_label += 1

    print(len(training_images))
    print(len(testing_images))
    # 随机打乱训练数据
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    state = np.random.get_state()
    np.random.shuffle(testing_images)
    np.random.set_state(state)
    np.random.shuffle(testing_labels)

    return np.asarray([training_images, training_labels, testing_images, testing_labels])


def main():
    with tf.Session() as sess:
        processed_data = create_image_list(sess)
        np.save(OUTPUT_FILE, processed_data)


if __name__ == '__main__':
    main()


