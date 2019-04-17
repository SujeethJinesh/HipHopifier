import time

from keras import backend as K, Model, Input
import numpy as np
from keras.layers import Conv1D, regularizers
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf

tf_session = K.get_session()
targetHeight = 640
targetWidth = 128
targetSize = (0, targetHeight, targetWidth)


def get_feature_reps(x, layer_names, model):
    """
    Get feature representations of input x for one or more layers in a given model.
    """
    featMatrices = []
    for ln in layer_names:
        import ipdb; ipdb.set_trace()
        selectedLayer = model.get_layer(ln)
        featRaw = selectedLayer.output
        featRawShape = K.shape(featRaw).eval(input=x, session=tf_session)
        N_l = featRawShape[-1]
        M_l = featRawShape[1] * featRawShape[2]
        featMatrix = K.reshape(featRaw, (M_l, N_l))
        featMatrix = K.transpose(featMatrix)
        featMatrices.append(featMatrix)
    return featMatrices


def get_content_loss(F, P):
    cLoss = 0.5 * K.sum(K.square(F - P))
    return cLoss


def get_Gram_matrix(F):
    G = K.dot(F, K.transpose(F))
    return G


def get_style_loss(ws, Gs, As):
    sLoss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]
        G_gram = get_Gram_matrix(G)
        A_gram = get_Gram_matrix(A)
        sLoss += w * 0.25 * K.sum(K.square(G_gram - A_gram)) / (N_l ** 2 * M_l ** 2)
    return sLoss


def get_total_loss(gImPlaceholder, cLayerName, sLayerNames, P, ws, As, gModel, alpha=1.0, beta=10000.0):
    F = get_feature_reps(gImPlaceholder, layer_names=[cLayerName], model=gModel)[0]
    Gs = get_feature_reps(gImPlaceholder, layer_names=sLayerNames, model=gModel)
    contentLoss = get_content_loss(F, P)
    styleLoss = get_style_loss(ws, Gs, As)
    totalLoss = alpha * contentLoss + beta * styleLoss
    return totalLoss


def calculate_loss(gImArr, gModel, cLayerName, sLayerNames, P, ws, As):
    """
    Calculate total loss using K.function
    """
    if gImArr.shape != (0, 640, 128):
        gImArr = gImArr.reshape((1, 0, 640, 128))
    loss_fcn = K.function([gModel.input], [get_total_loss(gModel.input, cLayerName, sLayerNames, P, ws, As)])
    return loss_fcn([gImArr])[0].astype('float64')


def get_grad(gImArr, gModel, cLayerName, sLayerNames, P, ws, As):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if gImArr.shape != (0, 640, 128):
        gImArr = gImArr.reshape((0, 640, 128))
    grad_fcn = K.function([gModel.input],
                          K.gradients(get_total_loss(gModel.input, cLayerName, sLayerNames, P, ws, As), [gModel.input]))
    grad = grad_fcn([gImArr])[0].flatten().astype('float64')
    return grad


def postprocess_array(x):
    # Zero-center by mean pixel
    if x.shape != (targetWidth, targetHeight, 3):
        x = x.reshape((targetWidth, targetHeight, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x


def style_transfer(model, content_audio, style_audio):
    gIm0 = np.random.randint(256, size=(0, 640, 128)).astype('float64')
    gIm0 = np.expand_dims(gIm0, axis=0)
    gImPlaceholder = K.placeholder(shape=(0, 640, 128))

    content_audio_tensor = tf.convert_to_tensor(content_audio, dtype='float32')
    # content_audio_tensor = Input(tensor=content_audio_tensor, shape=targetSize)

    style_audio_tensor = tf.convert_to_tensor(style_audio, dtype='float32')
    # style_audio_tensor = Input(tensor=style_audio_tensor, shape=targetSize)

    gImPlaceholder_tensor = tf.convert_to_tensor(gImPlaceholder, dtype='float32')
    # gImPlaceholder_tensor = Input(tensor=gImPlaceholder_tensor, shape=targetSize)

    # remove previous input layer
    model.layers.pop(0)

    FILTER_LENGTH = 5
    CONV_FILTER_COUNT = 56
    L2_regularization = 0.001

    n_features = content_audio_tensor.shape[2]
    input_shape = (None, n_features)
    model_input = Input(input_shape, name='input')

    model_input = Conv1D(
        filters=CONV_FILTER_COUNT,
        kernel_size=FILTER_LENGTH,
        kernel_regularizer=regularizers.l2(L2_regularization),  # Tried 0.001
        name='convolution_' + str(1)
    )(model_input)

    # import ipdb; ipdb.set_trace()

    # cModel = model(inputs=content_audio_tensor)
    # sModel = model(inputs=style_audio_tensor)
    # gModel = model(inputs=gImPlaceholder_tensor)

    cModel = model
    sModel = model
    gModel = model

    cLayerName = 'convolution_3'
    sLayerNames = [
        'convolution_1',
        'convolution_2',
        'convolution_3',
    ]

    P = get_feature_reps(x=content_audio_tensor, layer_names=[cLayerName], model=cModel)[0]
    As = get_feature_reps(x=style_audio_tensor, layer_names=sLayerNames, model=sModel)
    ws = np.ones(len(sLayerNames)) / float(len(sLayerNames))

    iterations = 600
    x_val = gIm0.flatten()
    start = time.time()

    # MIGHT HAVE SCOPING ISSUES HERE
    xopt, f_val, info = fmin_l_bfgs_b(calculate_loss(gImPlaceholder, gModel, cLayerName, sLayerNames, P, ws, As), x_val,
                                      fprime=get_grad(gImPlaceholder, gModel, cLayerName, sLayerNames, P, ws, As),
                                      maxiter=iterations, disp=True)
    xOut = postprocess_array(xopt)

    print('Image saved')
    end = time.time()
    print('Time taken: {}'.format(end - start))

    import ipdb;
    ipdb.set_trace()
