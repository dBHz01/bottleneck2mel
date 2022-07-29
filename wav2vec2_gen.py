import os
import torch
import librosa
import numpy as np
from fairseq import checkpoint_utils
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DATA_DIR = "./train_normalized/"
SAVE_DATA_DIR = "./train_normalized_processed/"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

models, _, _ = checkpoint_utils.load_model_ensemble_and_task(["./chinese-wav2vec2-base.pt"])
model = models[0]
model.eval()

filenames = []
text_content = {}

if __name__ == "__main__":
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
        filepath = DATA_DIR + speaker + "/" + f + ".wav"
        audio, sr = librosa.load(filepath, sr=16000, mono=True)
        audio = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.no_grad():
            vec, indexs = model.quantize(audio)
            vec = vec.numpy()
            indexs = indexs.numpy()
            if not os.path.exists(SAVE_DATA_DIR + speaker + "/"):
                os.mkdir(SAVE_DATA_DIR + speaker + "/")
            np.savez(SAVE_DATA_DIR + speaker + "/" + f, vec=vec, indexs=indexs)
    