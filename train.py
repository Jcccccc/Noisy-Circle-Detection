import numpy as np
import matplotlib.pyplot as plt
from utils import noisy_circle, iou
from keras import backend
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from model import SimpleCNNModel, BranchCNNModel, SequentialCNNModel

num_data = 80000
img_size = 200
max_radius = 50
noise_level = 2
train_ratio = 0.9
data_file_path = None

def circle_loss(y_true, y_pred):
    loss = abs(y_true[:, 0]-y_pred[:, 0]) * abs(y_true[:, 1]-y_pred[:, 1]) \
           + (y_true[:, 2]-y_pred[:, 2])**2
    return loss

def get_data(data_file=None, save_data=False):
    if data_file is not None:
        with load(data_file) as dataframe:
            data = dataframe['img']
            circles = dataframe['circles']
    else:
        data = np.zeros((num_data, img_size, img_size))
        circles = np.zeros((num_data, 3))
        for i in range(num_data):
            (row, col, rad), img = noisy_circle(img_size, max_radius, noise_level)
            data[i, :] = img / (noise_level+1)
            circles[i, :] = [row/img_size, col/img_size, rad/max_radius]
        if save_data:
            np.savez('data.npz', img=data, circles=circles)
    return data, circles


# get train & val data
data, circles = get_data(data_file=data_file_path)
# split data into train and validation
data = np.reshape(data, (num_data, img_size, img_size, 1))
num_train = int(train_ratio * num_data)
data_train, data_val = data[0:num_train], data[num_train:]
circles_train, circles_val = circles[0:num_train], circles[num_train:]

model = SequentialCNNModel()
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss=circle_loss, optimizer=opt)
'''
model.fit([data_train, data_train, data_train], circles_train, 
          batch_size=32, epochs=30,
          validation_data=([data_val, data_val, data_val], circles_val))
'''
# checkpoint
model_path = './models/sequential_model_pool5_5c_32_72.hdf5'
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')
history = model.fit(data_train, circles_train, 
          batch_size=32, epochs=30, callbacks=[checkpoint],
          validation_data=(data_val, circles_val))

# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('train_history.png')
