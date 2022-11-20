# Phrase break prediction for Text-to-Speech systems

This repository contains code to train speaker independent phrasing models for English Text-to-Speech systems. Currently the codebase supports two models

    1. BLSTM token classification model using task specific word embeddings trained from scratch
    2. Fine tuned BERT model with a token classification head  

The models are trained using the LibriTTS alignments available at [kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel). The `train-clean-360` split is used for training, while the `dev-clean` and `test-clean` splits are used for validation and test respectively.

# Quick start
## Download and preprocess the dataset
1. Download the dataset [kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel)

2. Preprocess the downloaded LibriTTS Label dataset and transform to a format suitable for the model

    ```python
    python utils/build_libritts_label_dataset.py \
        --dataset_dir <Path to the downloaded dataset> \
        --output_dir <Output dir, where the transformed dataset will be written>
    ```
## BLSTM token classification model using task specific word embeddings trained from scratch
### Train the model
1. Build vocabularies of words and tags from the processed dataset; for training word emebeddings from scratch

    ```python
    python utils/build_vocab_blstm.py \
        --dataset_dir <Directory containing the processed dataset>
    ```

    Running this script will save vocabulary files `dataset_dir/vocab/words.txt` and `dataset_dir/vocab/puncs.txt` containing all the words and punctuations in the dataset. It will also save `dataset_dir/vocab/params.json` with some extra information.

2. All model parameters as well as training hyperparameters are specified in `config/blstm.json`, which looks like

    ```json
    {
        "embedding_dim": 300,
        "num_blstm_layers": 2,
        "blstm_layer_size": 512,
        "batch_size": 64,
        "lr": 1e-3,
        "num_epochs": 10
    }
    ```
  To experiment with different values for model parameters/training hyperparameters, this file will have to be modified.

<!--
3. Train the model

    ```python
    python word_embedding_blstm_train.py \
        --config_file <path to config.json> \
        --data_dir <Directory containing the processed dataset> \
        --expereiment_dir <Directory where training artifacts will be saved> \
        --resume_checkpoint_path <If specified, load specified checkpoint and resume training>
    ```

4. Evaluate the model on the heldout test set

    ```python
    python word_embedding_blstm_evaluate.py \
        --config_file <path to config.json> \
        --vocab_dir <Directory containing the vocab files> \
        --test_data_dir <Directory containing the heldout test set> \
        --model_checkpoint <Trained model checkpoint to use for eval>
    ```
-->
## References
1. [Phrase break prediction with bidirectional encoder representations in Japanese text-to-speech synthesis](https://arxiv.org/pdf/2104.12395.pdf)
2. [An investigation of recurrent neural network architectures using word embeddings for phrase break prediction](https://www.isca-speech.org/archive_v0/Interspeech_2016/pdfs/0885.PDF)
