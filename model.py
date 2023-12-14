from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

height = 48
length = 48
channels = 1

def cnn_model_final_1():
    model = Sequential()
    # block 1
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(height, length, channels)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # block 2
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    # block 3
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    # block 4
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    # block 5
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    # dense layers
    model.add(Flatten())
    # block 6
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # block 7
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # block 8
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # block 9
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    # block 10
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # classification block
    model.add(Dense(4))
    model.add(Activation('softmax'))
    return model

def Initialize_model():
    model = cnn_model_final_1()
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])
    # load weights into new model
    model.load_weights("model_final_greyscale-1.h5")
    model.compile()
    return model