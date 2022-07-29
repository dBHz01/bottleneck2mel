import torch
import librosa
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

SAVE_DATA_DIR = "./train_normalized_processed/"
DATA_DIR = "./train_normalized/"

def gen_Mel(filename, speaker):
    ## return a numpy list
    y, sr = librosa.load(DATA_DIR + speaker + "/" + filename + ".wav", sr=16000)
    # transform = torchaudio.transforms.MelSpectrogram(sr, n_fft=400, win_length=400, hop_length=160)
    # mel_specgram = transform(y)
    n_fft = 400
    hop_length = 160 # wav2vec2模型的帧移是320个采样点(20ms), 那么在bottleneck2mel模型中，将输入的bottleneck特征长度乘以2，也就是每一帧重复一遍
    win_length = 400
    mel_specgram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return mel_specgram

def gen_std_Mel(filename, speaker):
    mel_mean = np.load(DATA_DIR + "mean.npy").reshape(-1, 1)
    mel_std = np.load(DATA_DIR + "std.npy").reshape(-1, 1)
    mel_specgram = gen_Mel(filename, speaker)
    mel_mean = np.repeat(mel_mean, mel_specgram.shape[1], axis=1)
    mel_std = np.repeat(mel_std, mel_specgram.shape[1], axis=1)
    return (mel_specgram - mel_mean) / mel_std
    
def cal_mean_and_std(speaker_range=range(2, 22)):
    # speaker_range contains the chosen speaker id
    filenames = []
    text_content = {}
    mels = []
    with open("aishell_transcript_v0.8.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in lines:
            i = i.replace("\n", "").split(" ")
            filename = i[0]
            i.remove(filename)
            filenames.append(filename)
            text_content[filename] = i
    for f in filenames:
        speaker = f[6:11]
        if int(speaker[2:]) not in speaker_range:
            continue
        data = np.load(SAVE_DATA_DIR + speaker + "/" + f + ".npz")
        data = data['vec']
        mels.append(gen_Mel(f, speaker))
    
    whole_mel = np.concatenate(mels, axis=1)
    print(whole_mel.shape)
    mel_mean = np.mean(whole_mel, axis=1)
    mel_std = np.std(whole_mel, axis=1)
    np.save(SAVE_DATA_DIR + "mean", mel_mean)
    np.save(SAVE_DATA_DIR + "std", mel_std)

if __name__ == "__main__":
    # mel_specgram = gen_Mel("BAC009S0002W0122", "S0002")
    std_mel_specgram = gen_std_Mel("BAC009S0002W0122", "S0002")
    print("Shape of spectrogram:{}".format(std_mel_specgram.shape))