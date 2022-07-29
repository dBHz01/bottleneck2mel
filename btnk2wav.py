import os
import torch
import click
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from train import Bottleneck2MelModel, Bottleneck2MelModel2
from mel_gen import gen_std_Mel

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    learning_rate = checkpoint['learning_rate']
    iteration = checkpoint['iteration']

    print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def load_bottleneck(speaker, filename, pred=False):
    SAVE_DATA_DIR = "./train_normalized_processed/"
    PRED_BTNK_DIR = "./pred_btnk/"
    if not pred:
        data = np.load(SAVE_DATA_DIR + speaker + "/" + filename + ".npz")['vec']
    else:
        data = np.load(PRED_BTNK_DIR + filename + ".npy")
    return data

def mel2wav(mel, test_speaker, test_filename, output_dir):
    # mel should be numpy array
    n_fft = 400
    hop_length = 160
    win_length = 400
    audio = librosa.feature.inverse.mel_to_audio(mel, sr=16000, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_iter=32)
    sf.write(output_dir + test_speaker + "_" + test_filename +  '.wav', audio, 16000)


@click.command()
@click.argument('checkpoint', type=click.Path(exists=True))
@click.argument('v', type=click.INT)
@click.argument('output', type=click.STRING)
def main(checkpoint, v, output):
    DATA_DIR = "./train_normalized/"
    EMBEDDING_DIM = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learning_rate = 1e-3
    test_speaker_id = 2
    test_speaker = "S0002"
    test_filename = "BAC009S0002W0122"
    if not os.path.exists(output):
        os.mkdir(output)
    if (v == 1):
        model = Bottleneck2MelModel(
            30, EMBEDDING_DIM, 4, p_drop=0.3, output_size=128, attn_num=6)
    else:
        model = Bottleneck2MelModel2(
            30, EMBEDDING_DIM, head_num=4, p_drop=0.5, output_size=128)
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adagrad(
        params=model.parameters(),
        lr=learning_rate,
        initial_accumulator_value=1e-8,
    )
    model, optimizer, learning_rate, iteration = load_checkpoint(checkpoint, model, optimizer)
    model.eval()
    with torch.no_grad():
        bottleneck = load_bottleneck(test_speaker, test_filename, pred=False)
        # pad
        print(bottleneck.shape)
        pred_mel = model(torch.tensor(bottleneck), torch.LongTensor([test_speaker_id]))
        pred_mel = torch.transpose(pred_mel, 0, 1)
        actual_mel = gen_std_Mel(test_filename, test_speaker)
        print(pred_mel.shape)
        print(actual_mel.shape)
        if (pred_mel.shape[1] > actual_mel.shape[1]):
            pred_mel = pred_mel[:, :actual_mel.shape[1] - pred_mel.shape[1]]
        elif (pred_mel.shape[1] < actual_mel.shape[1]):
            actual_mel = actual_mel[:, :pred_mel.shape[1] - actual_mel.shape[1]]
        loss = criterion(pred_mel, torch.tensor(actual_mel))
        print("loss: ", loss.item())
        mel_mean = np.load(DATA_DIR + "mean.npy").reshape(-1, 1)
        mel_std = np.load(DATA_DIR + "std.npy").reshape(-1, 1)
        mel_mean = np.repeat(mel_mean, pred_mel.shape[1], axis=1)
        mel_std = np.repeat(mel_std, pred_mel.shape[1], axis=1)
        mel2wav((pred_mel.detach().numpy() * mel_std) + mel_mean, test_speaker, test_filename, output)

if __name__ == "__main__":
    main()