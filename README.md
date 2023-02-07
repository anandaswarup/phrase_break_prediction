# Phrase break prediction for Text-to-Speech systems

This repository contains code to train speaker independent phrasing models for English Text-to-Speech systems. In text, phrase breaks are usually represented by punctuation. Typically, Text-to-Speech systems insert phrase breaks in the synthesized speech whenever they encounter a comma in the text to be synthesized.

Currently the codebase supports two models

    1. BLSTM token classification model using task specific word embeddings trained from scratch
    2. Fine tuned BERT model with a token classification head

Given unpunctuated text as input, these models punctuate the text with commas, and the text with predicted commas is then passed to the Text-to-Speech system to be synthesized.

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
1. Build vocabularies of words and punctuations from the processed dataset; for training word emebeddings from scratch

    ```python
    python utils/build_vocab_blstm.py \
        --dataset_dir <Directory containing the processed dataset>
    ```

    Running this script will save vocabulary files `dataset_dir/vocab/words.txt` and `dataset_dir/vocab/puncs.txt` containing all the words and punctuations in the dataset. It will also save `dataset_dir/vocab/params.json` with some extra information.

2. All model parameters as well as training hyperparameters are specified in `config/blstm.json`, which looks like

    ```json
    {
        "embedding_dim": 50,
        "num_blstm_layers": 2,
        "blstm_layer_size": 512,
        "batch_size": 64,
        "lr": 1e-3,
        "num_epochs": 10
    }
    ```
  To experiment with different values for model parameters/training hyperparameters, this file will have to be modified.

3. Train the model

    ```python
    python train_blstm.py \
        --config_file <Path to file containing the model/training configuration to be loaded> \
        --dataset_dir <Directory containing the processed dataset> \
        --expereiment_dir <Directory where training artifacts will be saved>
    ```

4. Evaluate the model on the heldout test set

    ```python
    python eval_blstm.py \
        --config_file <Path to file containing the model/training configuration to be loaded> \
        --dataset_dir <Directory containing the processed dataset> \
        --model_checkpoint <Path to the checkpoint containing the trained model to be used for eval>
    ```

5. Generate text with punctuations using the trained model

    ```python
    python generate_blstm.py \
        --config_file <Path to file containing model configuration to be loaded> \
        --in_text_file <Path to text file containing unpunctuated text> \
        --vocab_dir <Directory containing vocab files used to train the model> \
        --model_checkpoint <Path to the checkpoint containing the trained model to be used for generation> \
        --out_text_file <Output file where punctuated text will be written>    
    ```
## References
1. [Phrase break prediction with bidirectional encoder representations in Japanese text-to-speech synthesis](https://arxiv.org/pdf/2104.12395.pdf)
2. [An investigation of recurrent neural network architectures using word embeddings for phrase break prediction](https://www.isca-speech.org/archive_v0/Interspeech_2016/pdfs/0885.PDF)
