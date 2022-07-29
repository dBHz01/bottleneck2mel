import os
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import random
from mel_gen import gen_std_Mel
from model import Bottleneck2MelModel, Bottleneck2MelModel2

SEED = 126
BATCH_SIZE = 16
EMBEDDING_DIM = 128
BOTTLE_NECK_SIZE = 256
learning_rate = 1e-3
SAVE_DATA_DIR = "./train_normalized_processed/"
OUTPUT_DIR = "./output_model/"
SPEAKER_RANGE = range(2, 3)
TRAIN_MAX_LABEL = 400


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def save_checkpoint(checkpoint_path, model, optimizer, learning_rate, iteration):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    learning_rate = checkpoint['learning_rate']
    iteration = checkpoint['iteration']

    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def get_dataloader(batch_size=BATCH_SIZE):
    filenames = []
    text_content = {}
    data_train = []
    data_train_val = []
    train_speaker_id = []
    data_valid = []
    data_valid_val = []
    valid_speaker_id = []

    with open("aishell_transcript_v0.8.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in lines:
            i = i.replace("\n", "").split(" ")
            filename = i[0]
            i.remove(filename)
            filenames.append(filename)
            text_content[filename] = i
    # load train & valid data
    for f in filenames:
        speaker = f[6:11]
        label = f[-3:]
        if int(speaker[2:]) not in SPEAKER_RANGE:
            continue
        data = np.load(SAVE_DATA_DIR + speaker + "/" + f + ".npz")
        data = data['vec']
        data_Mel = gen_std_Mel(f, speaker)
        if int(label) < TRAIN_MAX_LABEL:
            # load train data
            data_train.append(torch.FloatTensor(data)[0])
            data_train_val.append(torch.transpose(
                torch.FloatTensor(data_Mel), 0, 1))
            train_speaker_id.append(int(speaker[2:]))
        else:
            # load valid data
            data_valid.append(torch.FloatTensor(data)[0])
            data_valid_val.append(torch.transpose(
                torch.FloatTensor(data_Mel), 0, 1))
            valid_speaker_id.append(int(speaker[2:]))

    train_dataset = Data.TensorDataset(nn.utils.rnn.pad_sequence(
        data_train, batch_first=True), nn.utils.rnn.pad_sequence(data_train_val, batch_first=True), torch.LongTensor(train_speaker_id))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    valid_dataset = Data.TensorDataset(nn.utils.rnn.pad_sequence(
        data_valid, batch_first=True), nn.utils.rnn.pad_sequence(data_valid_val, batch_first=True), torch.LongTensor(valid_speaker_id))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print("load data done")
    # TODO load test data
    # testLoader = torch.utils.data.DataLoader(
    #     datatest, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, valid_loader


def train(model,
          optimizer,
          train_loader,
          criterion):

    avg_loss = []
    print("begin training")
    model.train()

    for i, (data, val, speaker_id) in enumerate(train_loader):

        pred = model(data, speaker_id)
        if (pred.shape[1] > val.shape[1]):
            pred = pred[:, :val.shape[1] - pred.shape[1], :]
        elif (pred.shape[1] < val.shape[1]):
            val = val[:, :pred.shape[1] - val.shape[1], :]
        loss = criterion(pred, val)
        avg_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = np.array(avg_loss).mean()
    return avg_loss


def valid(model,
          optimizer,
          valid_loader,
          criterion):

    avg_loss = []
    print("begin validation")
    model.eval()
    with torch.no_grad():

        for i, (data, val, speaker_id) in enumerate(valid_loader):
            pred = model(data, speaker_id)
            if (pred.shape[1] > val.shape[1]):
                pred = pred[:, :val.shape[1] - pred.shape[1], :]
            elif (pred.shape[1] < val.shape[1]):
                val = val[:, :pred.shape[1] - val.shape[1], :]
            loss = criterion(pred, val)
            avg_loss.append(loss.item())

        avg_loss = np.array(avg_loss).mean()
        return avg_loss


if __name__ == "__main__":
    print("running on device: {}".format(device))
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    model = Bottleneck2MelModel(
        30, EMBEDDING_DIM, head_num=4, p_drop=0.3, output_size=128)
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
    )
    train_loader, valid_loader = get_dataloader()

    epoch = 10
    for epoch_i in range(epoch):

        print("Epoch: {}".format(epoch_i))

        start_time = time.time()

        train_loss = train(model, optimizer=optimizer, train_loader=train_loader,
                           criterion=criterion)

        end_time = time.time()
        print("train_loss: {} duration: {}".format(
            train_loss, end_time - start_time))

        print("Validation loss {}: {:9f}  ".format(
            epoch_i, valid(model, optimizer, valid_loader, criterion)))

        checkpoint_path = os.path.join(
            OUTPUT_DIR, "checkpoint_{}".format(epoch_i))
        save_checkpoint(checkpoint_path, model,
                        optimizer, learning_rate, epoch_i)
