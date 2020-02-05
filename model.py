from keras.models import Sequential, Model
from keras.layers import *

def SequentialCNNModel():
    model = Sequential()
    model.add(MaxPool2D(pool_size=(5, 5), input_shape = (200, 200, 1)))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                     activation ='relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                     activation ='relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                     activation ='relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), 
                     activation ='relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3),
                     activation ='relu'))
    #model.add(MaxPool2D(2, 2))
    model.add(Flatten())
    model.add(Dense(72))
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(3))
    return model

def BranchCNNModel():
    inputs = []
    conv_branches = []
    for conv_layer in [3, 4, 5]:
        input_img = Input(shape=(200, 200, 1))
        prep = MaxPool2D(pool_size=(5, 5))(input_img)
        conv = Conv2D(filters=32, kernel_size=(3, 3), activation ='relu')(prep)
        #bn = BatchNormalization()(conv)
        for _ in range(conv_layer-1):
            conv = Conv2D(filters=32, kernel_size=(3, 3), activation ='relu')(conv)
            #bn = BatchNormalization()(conv)
        flat = Flatten()(conv)
        conv_branches.append(flat)
        inputs.append(input_img)

    flat_concat = concatenate(conv_branches)
    fc_1 = Dense(64, activation='relu')(flat_concat)
    #fc_2 = Dense(256, activation='relu')(fc_1)
    #fc_bn = BatchNormalization()(fc_1)
    result = Dense(3)(fc_1)
    model = Model(inputs=inputs, outputs=result)

    return model
