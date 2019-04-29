import os
import numpy as np
import librosa
import librosa.display
import pickle
import matplotlib.pyplot as plt

import cv2
import argparse
import subprocess as sub

dict_genres = {'Electronic': 1, 'Experimental': 2, 'Folk': 3, 'Hip-Hop': 4,
               'Instrumental': 5, 'International': 6, 'Pop': 7, 'Rock': 8}


def get_tids_from_directory(audio_dir):
    """Get track IDs from the mp3s in a directory.
    Parameters
    ----------
    audio_dir : str
        Path to the directory where the audio files are stored.
    Returns
    -------
        A list of track IDs.
    """
    tids = []
    for _, dirnames, files in os.walk(audio_dir):
        if dirnames == []:
            tids.extend(int(file[:-4]) for file in files)
    return tids


def create_spectogram(audio_path, title):
    y, sr = librosa.load(audio_path)

    if title == "style_spectogram_stft_body_moving":
        import ipdb; ipdb.set_trace()

    D = np.log(np.abs(librosa.stft(y)) ** 2)[:, :434]

    D = np.stack([D, D, D], axis=2)

    D_min = np.min(D)
    D -= np.min(D)

    D_max = np.max(D)
    D *= 256.0 / np.max(D)

    cv2.imwrite('dataset/' + title + '.png', D)

    with open("dataset/" + title + "_min_max.pickle", "wb") as file:
        pickle.dump((D_min, D_max), file)

    return D
    # spect = librosa.feature.melspectrogram(S=D, n_mels=128, fmax=8000)
    # spect = librosa.feature.melspectrogram(S=D, fmax=8000)
    # spect = librosa.power_to_db(spect, ref=np.max)[:, :1702]
    # return spect.T


def autocrop_image(location):
    im = cv2.imread(location, cv2.IMREAD_UNCHANGED)

    y, x = im[:, :, 3].nonzero()  # get the nonzero alpha coordinates
    minx = np.min(x)
    miny = np.min(y)
    maxx = np.max(x)
    maxy = np.max(y)

    cropImg = im[miny:maxy, minx:maxx]

    cv2.imwrite(location, cropImg)


def plot_spect(spect, title):
    print(spect.shape)
    fig = plt.figure(frameon=False, figsize=(4.44444, 0.88888))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(spect.T, y_axis='log', cmap='gray_r', fmax=8000, x_axis='time')
    plt.margins(0)
    plt.axis('off')
    plt.savefig(title, bbox_inches='tight', transparent=True, pad_inches=0)
    autocrop_image(title)


def create_array_and_save(audio_path, title, spectogram_title, pickle_out):
    genres = []
    X_spect = np.empty((0, 1702, 128))
    spect = create_spectogram(audio_path, title)

    # plot_spect(spect, spectogram_title)
    #
    # # Normalize for small shape differences
    # spect = spect[:1702, :]
    # X_spect = np.append(X_spect[:1702, :], [spect], axis=0)
    #
    # if title == "content_audio":
    #     genres.append(dict_genres["Instrumental"])
    # else:
    #     genres.append(dict_genres["Hip-Hop"])
    #
    # y_arr = np.array(genres)
    # np.savez("dataset/" + title, X_spect, y_arr)
    #
    # pickle.dump(X_spect, pickle_out)
    #
    # return X_spect


def preprocess_audio():
    # import ipdb;
    # ipdb.set_trace()
    content_audio_title = "content_spectogram_stft_bossa_nova"
    content_audio_spectogram = "dataset/content_spectogram_stft_bossa_nova.png"
    content_audio_path = "dataset/content_audio/bensound-theelevatorbossanova.mp3"
    content_audio_spectogram_location = open("dataset/content_spectogram_stft.pickle", "wb")
    content_audio = create_array_and_save(content_audio_path, content_audio_title, content_audio_spectogram,
                                          content_audio_spectogram_location)

    style_audio_title = "style_spectogram_stft_body_moving"
    style_audio_spectogram = "dataset/style_spectogram_stft_body_moving.png"
    style_audio_path = "dataset/style_audio/gtzan_hiphop_10s.mp3"
    style_audio_spectogram_location = open("dataset/style_spectogram_stft.pickle", "wb")
    style_audio = create_array_and_save(style_audio_path, style_audio_title, style_audio_spectogram,
                                        style_audio_spectogram_location)

    return content_audio, style_audio


def spec_to_audio(spectrogram):
    # import ipdb; ipdb.set_trace()
    N_FFT = 798
    iters = 1000
    p = 2 * np.pi * np.random.random(spectrogram.shape) - np.pi
    for i in range(iters):
        S = np.exp(1j * p + spectrogram)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))
        # import ipdb; ipdb.set_trace()

    _, style_sr = librosa.load("dataset/content_audio/bensound-theelevatorbossanova.mp3")

    librosa.output.write_wav("outputs/output_new.wav", x, style_sr)


def deprocess_audio(filename):
    content_min, content_max = pickle.load(open("dataset/content_audio_min_max.pickle", "rb"))

    spectrogram = cv2.imread(filename)

    spectrogram = spectrogram * (content_max) / 256.0
    spectrogram += content_min

    spec_to_audio(spectrogram[:, :, 0])

    # import ipdb;
    # ipdb.set_trace()


parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=1, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')

preprocess_audio()
os.system("python3 keras_image_style_transfer_2.py dataset/content_spectogram_stft_bossa_nova.png dataset/style_spectogram_stft_body_moving.png outputs/output_new --iter 200")
deprocess_audio("outputs/output_new_at_iteration_199.png")
