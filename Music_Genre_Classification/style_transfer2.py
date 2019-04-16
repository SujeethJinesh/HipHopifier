from keras import Model, Input
from keras.layers import Conv1D, regularizers, BatchNormalization, Activation, MaxPooling1D, Dropout, LSTM, Dense
from keras.optimizers import Adam
from keras.models import load_model

from preprocess_audio import preprocess_audio

N_LAYERS = 3
FILTER_LENGTH = 5
CONV_FILTER_COUNT = 56
BATCH_SIZE = 32
LSTM_COUNT = 96
EPOCH_COUNT = 70
NUM_HIDDEN = 64
L2_regularization = 0.001
num_classes = 8


def load_model_for_style_transfer(model_input):
    # print('Building model...')
    # layer = model_input
    #
    # ### 3 1D Convolution Layers
    # for i in range(N_LAYERS):
    #     # give name to the layers
    #     layer = Conv1D(
    #         filters=CONV_FILTER_COUNT,
    #         kernel_size=FILTER_LENGTH,
    #         kernel_regularizer=regularizers.l2(L2_regularization),  # Tried 0.001
    #         name='convolution_' + str(i + 1)
    #     )(layer)
    #     layer = BatchNormalization(momentum=0.9)(layer)
    #     layer = Activation('relu')(layer)
    #     layer = MaxPooling1D(2)(layer)
    #     layer = Dropout(0.4)(layer)
    #
    # ## LSTM Layer
    # layer = LSTM(LSTM_COUNT, return_sequences=False)(layer)
    # layer = Dropout(0.4)(layer)
    #
    # ## Dense Layer
    # layer = Dense(NUM_HIDDEN, kernel_regularizer=regularizers.l2(L2_regularization), name='dense1')(layer)
    # layer = Dropout(0.4)(layer)
    #
    # ## Softmax Output
    # layer = Dense(num_classes)(layer)
    # layer = Activation('softmax', name='output_realtime')(layer)
    # model_output = layer
    # model = Model(model_input, model_output)
    #
    # opt = Adam(lr=0.001)
    # model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=opt,
    #     metrics=['accuracy']
    # )
    #
    # print(model.summary())
    # model.load_weights("models/crnn/")
    # return model

    model = load_model('models/crnn/weights.best.h5')  # load pre-trained model
    return model


if __name__ == '__main__':
    # Preprocess Audio Data
    preprocess_audio()

    # Load Model
    model = load_model_for_style_transfer(model_input)
