# Phrase break prediction using BLSTM

This repository contains code to train a phrase break prediction system for Text-to-Speech systems using BLSTMs and word embeddings. The sytem is trained used LibriTTS alignments provided at [kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel)
 

# Quick start
## Train model from scratch
1. Download the dataset [kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel)
    - We use `train-clean-360`, `dev-clean` and `test-clean` as our train, dev and test splits
    
2. Preprocess the downloaded LibriTTS Label dataset and save it in a convenient format

    ```python
    python build_LibriTTS_label_dataset.py \
        --raw_dataset_dir <Path to the downloaded dataset> \
        --processed_dataset_dir <Output dir, where the processed dataset will be written>
    ```


## References

1. [Phrase break prediction with bidirectional encoder representations in Japanese text-to-speech synthesis](https://arxiv.org/pdf/2104.12395.pdf)
2. [An investigation of recurrent neural network architectures using word embeddings for phrase break prediction](https://www.isca-speech.org/archive_v0/Interspeech_2016/pdfs/0885.PDF)
