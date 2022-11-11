# Phrase break prediction for Text-to-Speech systems

This repository contains code to train speaker independent phrasing models in English for Text-to-Speech systems. Input representations to the model are generated either from (1) Task specific word embeddings trained from scratch using a BLSTM language model, or (2) Fine-tuned BERT models. The models are trained using the LibriTTS alignments available at [kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel). The `train-clean-360` split is used for training, while the `dev-clean` and `test-clean` splits are used for validation and test respectively.

# Quick start
## Download and preprocess the dataset
1. Download the dataset [kan-bayashi/LibriTTSLabel](https://github.com/kan-bayashi/LibriTTSLabel)

2. Preprocess the downloaded LibriTTS Label dataset and transform to a format suitable for the model

    ```python
    python utils/build_libritts_label_dataset.py \
        --dataset_dir <Path to the downloaded dataset> \
        --output_dir <Output dir, where the transformed dataset will be written>
    ```
<!--
## Train Word Embedding + BLSTM model
1. Build vocabularies of words and tags from the processed dataset; for training word emebeddings from scratch

    ```python
    python utils/build_vocab_word_embeddings.py \
        --data_dir <Directory containing the processed dataset>
    ```

    Running this script will save vocabulary files `data_dir/vocab/words.txt` and `data_dir/vocab/tags.txt` containing all the words and tags in the dataset. It will also save `data_dir/vocab/dataset_params.json` with some extra information.

2. All model parameters as well as training hyperparameters are specified in `config/word_embedding_blstm_config.json`, which looks like

    ```json
    {
        "embedding_dim": 50,
        "blstm_size": 512,
        "batch_size": 64,
        "lr": 1e-5,
        "num_epochs": 50
    }
    ```
    To experiment with different values for model parameters/training hyperparameters, this file will have to be modified.

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
