from keras import Input, Model
from keras.models import load_model


def load_model_for_style_transfer():
    # load best weight model
    model = load_model('models/crnn/weights.best.h5')  # load pre-trained model

    # # might need to use this later for style transfer
    # newInput = Input(batch_shape=(0, 299, 299, 3))  # let us say this new InputLayer
    # newOutputs = model(newInput)
    # newModel = Model(newInput, newOutputs)
    # n_features = x_train.shape[2]
    # input_shape = (None, n_features)
    # model_input = Input(input_shape, name='input')

    return model
