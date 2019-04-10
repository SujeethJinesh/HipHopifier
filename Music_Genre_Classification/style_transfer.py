from keras import backend as K
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from music_tagger_crnn import MusicTaggerCRNN
from audio_conv_utils import preprocess_input, decode_predictions
import librosa
import librosa.display


def get_feature_reps(x, layer_names, model):
    """
    Get feature representations of input x for one or more layers in a given model.
    """
    featMatrices = []
    for ln in layer_names:
        selectedLayer = model.get_layer(ln)
        featRaw = selectedLayer.output
        featRawShape = K.shape(featRaw).eval(session=tf_session)
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


def get_total_loss(gImPlaceholder, alpha=1.0, beta=10000.0):
    F = get_feature_reps(gImPlaceholder, layer_names=[cLayerName], model=gModel)[0]
    Gs = get_feature_reps(gImPlaceholder, layer_names=sLayerNames, model=gModel)
    contentLoss = get_content_loss(F, P)
    styleLoss = get_style_loss(ws, Gs, As)
    totalLoss = alpha * contentLoss + beta * styleLoss
    return totalLoss


def calculate_loss(gImArr):
    """
    Calculate total loss using K.function
    """
    if gImArr.shape != (1, targetWidth, targetWidth, 3):
        gImArr = gImArr.reshape((1, targetWidth, targetHeight, 3))
    loss_fcn = K.function([gModel.input], [get_total_loss(gModel.input)])
    return loss_fcn([gImArr])[0].astype('float64')


def get_grad(gImArr):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if gImArr.shape != (1, targetWidth, targetHeight, 3):
        gImArr = gImArr.reshape((1, targetWidth, targetHeight, 3))
    grad_fcn = K.function([gModel.input],
                          K.gradients(get_total_loss(gModel.input), [gModel.input]))
    grad = grad_fcn([gImArr])[0].flatten().astype('float64')
    return grad


if __name__ == '__main__':
    content_audio_path = './dataset/content_audio/Dee_Yan-Key_-_01_-_Elegy_for_Argus.mov'
    style_audio_path = './dataset/style_audio/Yung_Kartz_-_02_-_Lethal.mov'
    output_path = './results/output.jpg'

    ## TODO: Fix this
    targetHeight = 512
    targetWidth = 512
    targetSize = (targetHeight, targetWidth)
    ##

    ## TODO: replace load_img with load_audio using librosa
    # content_audio = load_img(path=content_audio_path, target_size=targetSize)
    # content_audio_array = img_to_array(content_audio)
    # content_audio_array = K.variable(preprocess_input(np.expand_dims(content_audio_array, axis=0)), dtype='float32')
    content_audio = preprocess_input(content_audio_path)

    # import matplotlib.pyplot as plt

    # import ipdb; ipdb.set_trace()
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(content_audio,
    #                          y_axis='mel', fmax=8000, x_axis='time')
    #
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    # plt.tight_layout()

    # style_audio = load_img(path=style_audio_path, target_size=targetSize)
    # style_audio_array = img_to_array(style_audio)
    # style_audio_array = K.variable(preprocess_input(np.expand_dims(style_audio_array, axis=0)), dtype='float32')
    style_audio = preprocess_input(content_audio_path)

    gIm0 = np.random.randint(256, size=(targetWidth, targetHeight, 3)).astype('float64')
    gIm0 = np.expand_dims(gIm0, axis=0)
    gImPlaceholder = K.placeholder(shape=(1, targetWidth, targetHeight, 3))
    ##

    tf_session = K.get_session()

    cModel = MusicTaggerCRNN(weights='msd', input_tensor=content_audio, include_top=False)
    sModel = MusicTaggerCRNN(weights='msd', input_tensor=style_audio, include_top=False)
    gModel = MusicTaggerCRNN(weights='msd', input_tensor=gImPlaceholder, include_top=False)

    cLayerName = 'conv4'
    sLayerNames = [
        'conv1',
        'conv2',
        'conv3',
        'conv4',
    ]

    P = get_feature_reps(x=content_audio, layer_names=[cLayerName], model=cModel)[0]
    As = get_feature_reps(x=style_audio, layer_names=sLayerNames, model=sModel)
    ws = np.ones(len(sLayerNames)) / float(len(sLayerNames))

    iterations = 600
    x_val = gIm0.flatten()
    xopt, f_val, info = fmin_l_bfgs_b(calculate_loss, x_val, fprime=get_grad,
                                      maxiter=iterations, disp=True)
