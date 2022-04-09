import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
        x = dic["data".encode('utf-8')]
        y = dic["labels".encode('utf-8')]
    return x, y


def load_CIFAR(base_path, first_time=True):
    # x_train, x_test, y_train, y_test = None, None, None, None

    if first_time:
        x1, y1 = unpickle(base_path + "/data_batch_1")
        x2, y2 = unpickle(base_path + "/data_batch_2")
        x3, y3 = unpickle(base_path + "/data_batch_3")
        x4, y4 = unpickle(base_path + "/data_batch_4")
        x5, y5 = unpickle(base_path + "/data_batch_5")
        x = np.concatenate((x1, x2, x3, x4, x5), axis=0)
        y = np.concatenate((y1, y2, y3, y4, y5), axis=0)

        x = np.reshape(x, (50000, 3, 32, 32)).transpose((0, 2, 3, 1))
        x = x.astype('float32') / 255.0
        y = to_categorical(y, num_classes=10)       # one-hot encode for labels

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        np.save(base_path + "/x_train", x_train)
        np.save(base_path + "/y_train", y_train)
        np.save(base_path + "/x_test", x_test)
        np.save(base_path + "/y_test", y_test)

        # # view the image
        # import cv2
        # for i in range(10):
        #     test = x_train[i]
        #     test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
        #     test = cv2.resize(test, (128, 128))     # raw size is 32*32
        #     cv2.imshow("test", test)
        #     cv2.waitKey(1000)

    else:
        x_train = np.load(base_path + "/x_train.npy")
        y_train = np.load(base_path + "/y_train.npy")
        x_test = np.load(base_path + "/x_test.npy")
        y_test = np.load(base_path + "/y_test.npy")

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    load_CIFAR("./cifar-10-data", True)
    print("Preprocessing done!")
