from tensorflow.keras import models, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, \
    Dropout, BatchNormalization, Flatten, add, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt
from processing_CIFAR import *


class CifarResNet:
    def __init__(self, if_load=False):
        self.x_train, self.x_test, self.y_train, self.y_test = load_CIFAR("cifar-10-data")

        self.batch_size = 128
        self.epoch = 2000

        self.network = self.setup_network()
        self.model_file = "cifar-0408.h5"

        if if_load:
            self.network.load_weights(self.model_file)

    def setup_network(self):
        def res_block(x0, filters):
            x1 = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(x0)
            x1 = BatchNormalization()(x1)
            x2 = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')(x1)
            x2 = BatchNormalization()(x2)
            y = add([x0, x2])
            return Activation('relu')(y)

        inputs = Input(shape=self.x_train.shape[1:])

        # 32*32*3
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        # 16*16*32
        x = res_block(x, 32)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        # 8*8*64
        x = res_block(x, 64)
        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        # 4*4*128
        x = res_block(x, 128)
        x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.2)(x)

        # 4*4*256
        x = res_block(x, 256)
        x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

        # 2*2*512
        x = res_block(x, 512)
        # x = res_block(x, 512)
        # x = res_block(x, 512)
        x = Conv2D(filters=1024, kernel_size=(2, 2), activation='relu', padding='valid')(x)
        x = Dropout(rate=0.2)(x)
        x = Flatten()(x)

        # 1*1*512
        x = Dense(units=512, kernel_initializer='normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(units=256, kernel_initializer='normal', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.2)(x)
        outputs = Dense(units=10, kernel_initializer='normal', activation='softmax')(x)

        network = models.Model(inputs=inputs, outputs=outputs)

        # different kinds of optimizers
        opt = Adam(lr=0.00005, decay=1e-6)
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

        f1 = open("loss.txt", 'w')
        for each in self.network.history.history['loss']:
            f1.write(str(each) + "\n")
        f1.close()
        f2 = open("val_loss.txt", 'w')
        for each in self.network.history.history['val_loss']:
            f2.write(str(each) + "\n")
        f2.close()

        plt.figure(0)
        plt.plot(self.network.history.history['loss'], 'r')
        plt.plot(self.network.history.history['val_loss'], 'g')
        plt.xticks(np.arange(250, 250 + self.epoch + 1, 10.0))
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

        f3 = open("final accuracy.txt", 'w')
        f3.write(str(correctness / len(result)))


if __name__ == '__main__':
    my_predictor = CifarResNet(if_load=False)
    my_predictor.training()
    my_predictor.evaluate()
