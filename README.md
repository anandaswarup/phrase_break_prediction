# Phrase break prediction using BLSTM

This repository contains code to train a phrase break prediction model for Text-to-Speech systems using BLSTMs and word embeddings. The sytem is trained using LibriTTS alignments provided at [kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel). The `train-clean-360` split is used for training, while the `dev-clean` and `test-clean` splits are used for validation and test respectively.
 

# Quick start
## Train model from scratch
1. Download the dataset [kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel)

2. Preprocess the downloaded LibriTTS Label dataset and save it in a convenient format

    ```python
    python build_LibriTTS_label_dataset.py \
        --raw_dataset_dir <Path to the downloaded dataset> \
        --processed_dataset_dir <Output dir, where the processed dataset will be written>
    ```

3. Build vocabularies and parameters for the dataset

    ```python
    python build_vocab \
        --data_dir <Directory containing the dataset>
    ```

    Running this script will generate vocabulary files `words.txt` and `tags.txt` containing all the words and tags in the dataset. It will also save `dataset_params.json` with some extra information.

4. All parameters/hyperparameters used to train the model are specified in `config.json`, which looks like

    ```json
    {
        "embedding_dim": 50,
        "blstm_size": 512,
        "batch_size": 64,
        "lr": 1e-5,
        "num_epochs": 50
    }
    ```
    In order to experiment with different values for parameters/hyperparameters you will have to modify the `config.json`.

## References

1. [Phrase break prediction with bidirectional encoder representations in Japanese text-to-speech synthesis](https://arxiv.org/pdf/2104.12395.pdf)
2. [An investigation of recurrent neural network architectures using word embeddings for phrase break prediction](https://www.isca-speech.org/archive_v0/Interspeech_2016/pdfs/0885.PDF)
