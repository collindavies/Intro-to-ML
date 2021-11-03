import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
from tensorflow.python.ops.gen_batch_ops import Batch

from deeplearning_models import functional_model, MyCustomModel
from my_utils import display_some_examples

# tensorflow.keras.Sequential - rarely used for complex deep learning tasks
seq_model = tensorflow.keras.Sequential(
    [
        Input(shape = (28, 28, 1)), ## Input needs to be 28 pixels by 28 pixels and 1 channel (gray-scale)
        Conv2D(32, (3, 3), activation = 'relu'),
        Conv2D(64, (3, 3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation = 'relu'),
        Dense(10, activation = 'softmax') ## Output model needs to have 10 outputs (desired outputs are 0-9)
    ]
)




if __name__=='__main__':
    

    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    if False:
        display_some_examples(x_train, y_train)

    x_train = x_train.astype('float32') / 255 ## Normalize function -> Float format so you can get decimal values. 0 equals white. 255 equals black.
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis = -1)
    x_test = np.expand_dims(x_test, axis = -1)

    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    # model = functional_model()
    model = MyCustomModel()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')

    # label : 2
    # one hot encoding : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] - required for categorical_crossentropy (not required for sparsecategorical_crossentropy)
    
    # Model training
    model.fit(x_train, y_train, batch_size = 64, epochs = 3, validation_split = 0.2)
    
    # Evaluation on test set
    model.evaluate(x_test, y_test, batch_size = 64)