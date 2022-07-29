# 暑期实践-声音复刻

本项目为声音复刻项目，参考复现文章《Cloning one's voice using very limited data in the wild》。本仓库为其中bottleneck2mel的部分。使用数据集为AI-SHELL-1。

### 仓库组成

- `train_normalized`（一致化后的音频，需自行添加）

- `train_normalized_processed`（音频经过wav2vec2后得到的bottleneck，需自行添加）

- `output_model`（模型路径，训练过程中会自行创建）

- `output_wav`（最终输出音频，需自行添加）

- `.gitignore`

- `aishell_transcript_v0.8.txt`（数据集对应文字）

- `btnk2wav.py`

  - 将bottleneck转化成音频。

  - 出于简便bottleneck使用了S0002的BAC009S0002W0122音频的bottleneck，可自行更改。

  - 运行方式：`python ./btnk2wav.py ${model} 1 ./output_wav/`

    其中`${model}`为选用的模型路径，1表示使用模型`Bottleneck2MelModel`，2表示使用模型`Bottleneck2MelModel2`

- `chinese-wav2vec2-base.pt`，`chinese-wav2vec2-large.pt`（两个wav2vec2的预训练模型，需自行添加）

- `codebook.npy`

- `mel_gen.py`

  - 通过路径生成mel图。
  - 运行方式：`python ./mel_gen.py`
  - 其中`gen_Mel`函数生成标准mel图，`gen_std_Mel`根据计算好的平均值和标准差一致化后再生成mel图。

- `model.py`（两个模型）

- `README.md`

- `requirements.txt`

- `train.py`（训练脚本，可直接运行）

- `wav2vec2_gen.py`（音频生成bottleneck。可直接运行）

- `wav2vecTool.py`（音频转bottleneck）

