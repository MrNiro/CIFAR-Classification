from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, \
    Dropout, BatchNormalization, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
from processing_CIFAR import *


class CifarVGG:
    def __init__(self, if_load=False):
        self.x_train, self.x_test, self.y_train, self.y_test = load_CIFAR("../cifar-10-data")

        self.batch_size = 32
        self.epoch = 250
        self.network = self.setup_network()
        self.model_file = "cifar-0407.h5"

        if if_load:
            self.network.load_weights(self.model_file)

    def setup_network(self):
        network = models.Sequential()  # create sequential multi-layer perceptron

        # 32*32*3
        network.add(Conv2D(filters=32, input_shape=self.x_train.shape[1:],
                                  kernel_size=(3, 3), activation='relu', padding='same'))
        network.add(BatchNormalization())
        network.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
        network.add(BatchNormalization())
        network.add(MaxPooling2D(pool_size=(2, 2)))
        network.add(Dropout(rate=0.2))

        # 16*16*32
        network.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        network.add(BatchNormalization())
        network.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        network.add(BatchNormalization())
        network.add(MaxPooling2D(pool_size=(2, 2)))
        network.add(Dropout(rate=0.2))

        # 8*8*64
        network.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        network.add(BatchNormalization())
        network.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        network.add(BatchNormalization())
        network.add(MaxPooling2D(pool_size=(2, 2)))
        network.add(Dropout(rate=0.2))

        # 4*4*128
        network.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
        network.add(BatchNormalization())
        network.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
        network.add(BatchNormalization())
        network.add(MaxPooling2D(pool_size=(2, 2)))
        network.add(Dropout(rate=0.2))

        # 2*2*256
        network.add(Conv2D(filters=512, kernel_size=(2, 2), activation='relu', padding='same'))
        network.add(BatchNormalization())
        network.add(Conv2D(filters=512, kernel_size=(2, 2), activation='relu', padding='same'))
        network.add(BatchNormalization())

        # use valid convolutional to replace last MaxPooling
        network.add(Conv2D(filters=512, kernel_size=(2, 2), activation='relu', padding='valid'))
        network.add(BatchNormalization())
        network.add(Dropout(rate=0.2))

        # 1*1*512
        network.add(Flatten())

        network.add(Dense(units=512, kernel_initializer='normal', activation='relu'))
        network.add(BatchNormalization())
        network.add(Dropout(rate=0.2))

        network.add(Dense(units=256, kernel_initializer='normal', activation='relu'))
        network.add(BatchNormalization())
        network.add(Dropout(rate=0.2))

        network.add(Dense(units=64, kernel_initializer='normal', activation='relu'))
        network.add(BatchNormalization())
        network.add(Dropout(rate=0.2))

        network.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

        # different kinds of optimizers
        opt = Adam(lr=0.0001, decay=1e-5)
        network.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return network

    def training(self):
        # data augmentation
        data_generator = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        data_generator.fit(self.x_train)

        self.network.fit_generator(data_generator.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                                   epochs=self.epoch, validation_data=(self.x_test, self.y_test),
                                   verbose=1, shuffle=True)
        self.network.save_weights(self.model_file)

        plt.figure(0)
        plt.plot(self.network.history.history['loss'], 'r')
        plt.plot(self.network.history.history['val_loss'], 'g')
        plt.xticks(np.arange(0, self.epoch + 1, 5.0))
        plt.rcParams['figure.figsize'] = (28, 6)
        plt.xlabel("Num of Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Validation Loss")
        plt.legend(['train', 'validation'])
        plt.savefig('training.png', dpi=600)
        plt.show()

    def evaluate(self):
        # inference time
        import time
        start_time = time.time()
        for i in range(10):
            self.network.predict(self.x_test)
        end_time = time.time()
        avg_time = (end_time - start_time) / 10 / self.x_test.shape[0] * 1000
        print("Inference time: %.2f ns" % avg_time)

        result = self.network.predict(self.x_test)

        correctness = 0
        for i in range(len(result)):
            pre = np.argmax(result[i])
            truth = np.argmax(self.y_test[i])
            if pre == truth:
                correctness += 1

        # # save network structure as an image
        plot_model(self.network, to_file="model.png", show_shapes=True)
        print("Accuracy = %.4f" % (correctness / len(result)))


if __name__ == '__main__':
    my_predictor = CifarVGG(if_load=True)
    my_predictor.training()
    my_predictor.evaluate()
