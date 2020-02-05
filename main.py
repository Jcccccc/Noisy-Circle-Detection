import numpy as np
from utils import noisy_circle, iou
from keras.optimizers import Adagrad, Adam, Adadelta, RMSprop
from model import SequentialCNNModel, BranchCNNModel

def find_circle(img):
    # Fill in this function
    model = SequentialCNNModel()
    model.load_weights('./models/sequential_model_pool5_5c_32_72.hdf5')
    img = np.reshape(img, (1, 200, 200, 1))
    img = img / 3
    pred = model.predict(img)[0]
    return pred[0]*200, pred[1]*200, pred[2]*50

def find_circle_batch(img_batch, model_type='branch'):
    # Function for detect circles in a batch of images
    img_batch = np.reshape(img_batch, (len(img_batch), 200, 200, 1)) / 3
    if model_type == 'sequential':
        model = SequentialCNNModel()
        model.load_weights('./models/sequential_model_pool5_5c_32_72.hdf5')
    elif model_type == 'branch':
        model = BranchCNNModel()
        model.load_weights('./models/branch_model_pool5_345_64.h5')
        img_batch = [img_batch, img_batch, img_batch]
    else:
        return None
    preds = model.predict(img_batch)
    ret = [(params[0]*200, params[1]*200, params[2]*50) for params in preds]
    return ret

def main():
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())

def main_batch():
    # get results with prediction on batch
    results = []
    params_batch, img_batch = [], []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        params_batch.append(params)
        img_batch.append(img)
    detected = find_circle_batch(img_batch)
    results = np.array([iou(params_batch[i], detected[i]) for i in range(1000)])
    print((results > 0.7).mean())

if __name__ == '__main__':
    main()