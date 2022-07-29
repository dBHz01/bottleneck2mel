import torch
import numpy as np
import click
from fairseq import checkpoint_utils
import soundfile
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

@click.command()
@click.argument('wav', type=click.Path(exists=True))
@click.argument('model', type=click.Path(exists=True))
def main(wav, model):
    '''
    python <this script> wav model
    '''

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([model])
    model = models[0]
    model.eval()

    # 如何导出模型的码本
    # codebook = model.quantizer.vars.data  # (1, 640, 256)
    # codebook = codebook.view(model.quantizer.groups, model.quantizer.num_vars, -1)  # (2, 320, 256)
    # np.save('./codebook.npy', codebook.cpu().numpy())

    outputpath = wav + '.npz'

    wav, sampleRate = soundfile.read(wav)
    assert sampleRate == 16000

    audio = torch.from_numpy(wav).float().unsqueeze(0)
    with torch.no_grad():
        vec, indexs = model.quantize(audio)
        vec = vec.numpy()
        indexs = indexs.numpy()
        np.savez(outputpath, vec=vec, indexs=indexs)


if __name__ == '__main__':
    main()
