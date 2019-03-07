import librosa
import matplotlib.pyplot as plt


def preprocess_audio(audio_path, filename):
    cmap = plt.get_cmap('inferno')
    y, sr = librosa.load(audio_path, mono=True, duration=5)
    plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
    plt.axis('off')
    plt.savefig(f'test_img_data/{filename}.png')
    plt.clf()
    return


# util function to convert a tensor into a valid audio file
# https://librosa.github.io/librosa/generated/librosa.core.istft.html
def deprocess_audio(y):
    D = librosa.stft(y)
    return librosa.istft(D)
