from sklearn.decomposition import FastICA
from scipy.io import wavfile
from plotnine import *
import pandas as pd
import numpy as np


def create_dataset(type):
    if type == "music":
        dim = 2650546
    elif type == "speech":
        dim = 889216
    else:
        dim = 0
    dataset = np.ndarray(shape=(dim, 4))
    for i in range(4):
        fs, source = wavfile.read("data/partB/" + type + "mix2019_0{}.wav".format(i+1))
        dataset[:, i] = source
    return pd.DataFrame(dataset, columns=["s1", "s2", "s3", "s4"])


def plot_waveforms(dataset):
    for i in range(4):
        vec = (ggplot(data=dataset, mapping=aes(x=dataset.index.values, y=dataset["s{}".format(i+1)].values)) +
               geom_line() +
               xlab("") +
               ylab("") +
               ggtitle("reconst {}".format(i+1)))
        print(vec)


if __name__ == '__main__':
    type = "speech"
    ds = create_dataset(type)
    # plot_waveforms(ds)
    ica = FastICA(n_components=4)
    reconst = ica.fit_transform(ds) * 1000
    wavfile.write("outputs/partB/" + type + "mix_source_01.wav", 44100, reconst[:, 0])
    wavfile.write("outputs/partB/" + type + "mix_source_02.wav", 44100, reconst[:, 1])
    wavfile.write("outputs/partB/" + type + "mix_source_03.wav", 44100, reconst[:, 2])
    wavfile.write("outputs/partB/" + type + "mix_source_04.wav", 44100, reconst[:, 3])
    plot_waveforms(pd.DataFrame(reconst, columns=["s1", 's2', 's3', 's4']))