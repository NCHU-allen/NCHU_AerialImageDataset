from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, concatenate, Input, AveragePooling2D, add, GlobalAveragePooling2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model

def VGG19(input_shape, output_class):
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
               activation='relu'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same', ),
        Conv2D(256, (3, 3), activation='relu', padding='same', ),
        Conv2D(256, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        Conv2D(512, (3, 3), activation='relu', padding='same', ),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(output_class, activation='softmax')
    ])
    model.summary()

    return model

def Alexnet(input_shape, output_class):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_class, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    return model

#Build LetNet model with Keras
# https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-cnn%E6%BC%94%E5%8C%96%E5%8F%B2-alexnet-vgg-inception-resnet-keras-coding-668f74879306
def LetNet(input_shape, output_class):
    # initialize the model
    model = Sequential()

    # first layer, convolution and pooling
    model.add(Conv2D(input_shape=input_shape, kernel_size=(5, 5), filters=6, strides=(1,1), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second layer, convolution and pooling
    model.add(Conv2D(input_shape=input_shape, kernel_size=(5, 5), filters=16, strides=(1,1), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connection layer
    model.add(Flatten())
    model.add(Dense(120,activation = 'tanh'))
    model.add(Dense(84,activation = 'tanh'))

    # softmax classifier
    model.add(Dense(output_class))
    model.add(Activation("softmax"))

    return model

# Build InceptionV1 model
def InceptionV1(input_shape, output_class):
    # Define convolution with batchnromalization
    def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    # Define Inception structure
    def Inception(x, nb_filter_para):
        (branch1, branch2, branch3, branch4) = nb_filter_para
        branch1x1 = Conv2D(branch1[0], (1, 1), padding='same', strides=(1, 1), name=None)(x)

        branch3x3 = Conv2D(branch2[0], (1, 1), padding='same', strides=(1, 1), name=None)(x)
        branch3x3 = Conv2D(branch2[1], (3, 3), padding='same', strides=(1, 1), name=None)(branch3x3)

        branch5x5 = Conv2D(branch3[0], (1, 1), padding='same', strides=(1, 1), name=None)(x)
        branch5x5 = Conv2D(branch3[1], (1, 1), padding='same', strides=(1, 1), name=None)(branch5x5)

        branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branchpool = Conv2D(branch4[0], (1, 1), padding='same', strides=(1, 1), name=None)(branchpool)

        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)

        return x

    inpt = Input(shape=input_shape)

    x = Conv2d_BN(inpt, 64, (7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Inception(x, [(64,), (96, 128), (16, 32), (32,)])  # Inception 3a 28x28x256
    x = Inception(x, [(128,), (128, 192), (32, 96), (64,)])  # Inception 3b 28x28x480
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 14x14x480

    x = Inception(x, [(192,), (96, 208), (16, 48), (64,)])  # Inception 4a 14x14x512
    x = Inception(x, [(160,), (112, 224), (24, 64), (64,)])  # Inception 4a 14x14x512
    x = Inception(x, [(128,), (128, 256), (24, 64), (64,)])  # Inception 4a 14x14x512
    x = Inception(x, [(112,), (144, 288), (32, 64), (64,)])  # Inception 4a 14x14x528
    x = Inception(x, [(256,), (160, 320), (32, 128), (128,)])  # Inception 4a 14x14x832
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 7x7x832

    x = Inception(x, [(256,), (160, 320), (32, 128), (128,)])  # Inception 5a 7x7x832
    x = Inception(x, [(384,), (192, 384), (48, 128), (128,)])  # Inception 5b 7x7x1024

    # Using AveragePooling replace flatten
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(output_class, activation='softmax')(x)

    model = Model(input=inpt, output=x)

    return model

# Built ResNet34
def ResNet34(input_shape, output_class):
    # Define convolution with batchnromalization
    def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    # Define Residual Block for ResNet34(2 convolution layers)
    def Residual_Block(input_model, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
        x = Conv2d_BN(input_model, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')

        # need convolution on shortcut for add different channel
        if with_conv_shortcut:
            shortcut = Conv2d_BN(input_model, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, input_model])
            return x

    Img = Input(shape=input_shape)

    x = Conv2d_BN(Img, 64, (7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Residual conv2_x ouput 56x56x64
    x = Residual_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=64, kernel_size=(3, 3))

    # Residual conv3_x ouput 28x28x128
    x = Residual_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2),
                       with_conv_shortcut=True)  # need do convolution to add different channel
    x = Residual_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=128, kernel_size=(3, 3))

    # Residual conv4_x ouput 14x14x256
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2),
                       with_conv_shortcut=True)  # need do convolution to add different channel
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3, 3))

    # Residual conv5_x ouput 7x7x512
    x = Residual_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Residual_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = Residual_Block(x, nb_filter=512, kernel_size=(3, 3))

    # Using AveragePooling replace flatten
    x = GlobalAveragePooling2D()(x)
    x = Dense(output_class, activation='softmax')(x)

    model = Model(input=Img, output=x)
    return model