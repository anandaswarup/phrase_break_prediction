# Phrase break prediction using BLSTM

This repository contains code to train a phrase break prediction model for Text-to-Speech systems using BLSTMs and word embeddings. The sytem is trained using LibriTTS alignments provided at [kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel). The `train-clean-360` split is used for training, while the `dev-clean` and `test-clean` splits are used for validation and test respectively.
 

# Quick start
## Train model from scratch
1. Download the dataset [kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel)

2. Preprocess the downloaded LibriTTS Label dataset and save it in a convenient format which will be used by the model later

    ```python
    python build_LibriTTS_label_dataset.py \
        --raw_dataset_dir <Path to the downloaded dataset> \
        --processed_dataset_dir <Output dir, where the processed dataset will be written>
    ```

3. Build vocabularies of words and tags from the processed dataset

    ```python
    python build_vocab \
        --data_dir <Directory containing the processed dataset>
    ```

    Running this script will save vocabulary files `data_dir/vocab/words.txt` and `data_dir/vocab/tags.txt` containing all the words and tags in the dataset. It will also save `data_dir/vocab/dataset_params.json` with some extra information.

4. All model parameters as well as training hyperparameters are specified in `config.json`, which looks like

    ```json
    {
        "embedding_dim": 50,
        "blstm_size": 512,
        "batch_size": 64,
        "lr": 1e-5,
        "num_epochs": 50
    }
    ```
    To experiment with different values for model parameters/training hyperparameters, `config.json` will have to be modified.

5. Train the model

    ```python
    python train.py \
        --config_file <path to config.json> \
        --data_dir <Directory containing the processed dataset> \
        --expereiment_dir <Directory where training artifacts will be saved> \
        --resume_checkpoint_path <If specified, load specified checkpoint and resume training>
    ```

6. Evaluate the model

    ```python
    python evaluate.py \
        --config_file <path to config.json> \
        --vocab_dir <Directory containing the vocab files> \
        --test_data_dir <Directory containing the test dataset> \
        --model_checkpoint <Trained model checkpoint to use for eval>
    ```

## References
1. [Phrase break prediction with bidirectional encoder representations in Japanese text-to-speech synthesis](https://arxiv.org/pdf/2104.12395.pdf)
2. [An investigation of recurrent neural network architectures using word embeddings for phrase break prediction](https://www.isca-speech.org/archive_v0/Interspeech_2016/pdfs/0885.PDF)
