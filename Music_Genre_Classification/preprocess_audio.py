import os
import numpy as np
import librosa
import librosa.display
import pickle
import matplotlib.pyplot as plt

import cv2

from PIL import Image

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
    D = np.log(np.abs(librosa.stft(y)) ** 2)[:, :640]

    D = np.stack([D, D, D], axis=2)

    D_min = np.min(D)
    D -= np.min(D)

    D_max = np.max(D)
    D *= 256.0 / np.max(D)

    # if title == "style_audio":
    #     import ipdb; ipdb.set_trace()

    cv2.imwrite('dataset/' + title + '.png', D)

    with open("dataset/" + title + "_min_max.pickle", "wb") as file:
        pickle.dump((D_min, D_max), file)

    return D
    # spect = librosa.feature.melspectrogram(S=D, n_mels=128, fmax=8000)
    # spect = librosa.feature.melspectrogram(S=D, fmax=8000)
    # spect = librosa.power_to_db(spect, ref=np.max)[:, :640]
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
    X_spect = np.empty((0, 640, 128))
    spect = create_spectogram(audio_path, title)

    # plot_spect(spect, spectogram_title)
    #
    # # Normalize for small shape differences
    # spect = spect[:640, :]
    # X_spect = np.append(X_spect[:640, :], [spect], axis=0)
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
    content_audio_title = "content_audio"
    content_audio_spectogram = "dataset/content_spectogram_stft.png"
    content_audio_path = "dataset/content_audio/Dee_Yan-Key_-_01_-_Elegy_for_Argus.mov"
    content_audio_spectogram_location = open("dataset/content_spectogram_stft.pickle", "wb")
    content_audio = create_array_and_save(content_audio_path, content_audio_title, content_audio_spectogram,
                                          content_audio_spectogram_location)

    style_audio_title = "style_audio"
    style_audio_spectogram = "dataset/style_spectogram_stft.png"
    style_audio_path = "dataset/style_audio/Yung_Kartz_-_02_-_Lethal.mp3"
    style_audio_spectogram_location = open("dataset/style_spectogram_stft.pickle", "wb")
    style_audio = create_array_and_save(style_audio_path, style_audio_title, style_audio_spectogram,
                                        style_audio_spectogram_location)

    return content_audio, style_audio


preprocess_audio()
