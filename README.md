# Noisy Circle Detection

### Files
- **main.py**: Used for prediction and get AP@0.7 result. Besides the `main` function provided, I also implement a `main_batch` function to predict all test inputs as a batch to significantly improve the test speed.
- **train.py**: Used for train models. It will save best checkpoint with lowest validation loss (val_loss).
- **model.py**: Defined two structures of CNN models I tested for this task.
- **utils.py**: Some utility functions. To keep a better project structure, I move some functions (`iou`, `noisy_circle`) originally in main.py to this file.

### Environment
I used keras v2.2.5 with Tensorflow v1.14.0 as backend.

### Assumptions
As the `main` function in `main.py` has indicated that the test inputs are all with row = col = 200 and rad = 50. So I assume all the data follow these parameters and generate training and validation data.

### Notes
The final results with my submitted models are ~0.84 for sequential model and ~0.87 for branch model. Due to the limitation of my computation resources, I can only do experiments with not too many Conv layers and not too many training data. I definitely believe if I have more time and resouces for experiments, I can get better structures and better results.