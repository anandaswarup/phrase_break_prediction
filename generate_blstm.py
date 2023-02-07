"""Generate text with punctuations using the trained BLSTM token classification model"""

import argparse
import os
import torch
from model.blstm import PhraseBreakPredictor
from utils.utils import load_checkpoint_to_evaluate_model, load_json_to_dict, load_vocab_to_dict


def load_text_instances(synthesis_file):
    """Load text to be punctuated from file"""
    with open(synthesis_file, "r") as file_reader:
        text_instances = file_reader.readlines()

    text_instances = [instance.strip("\n") for instance in text_instances]
    text_instances = [instance.split("|") for instance in text_instances]

    return text_instances


def load_vocabs(vocab_dir):
    """Load the vocab files for the BLSTM token classification model"""

    assert os.path.isdir(vocab_dir), f"Vocab dir does not exist, run build_vocab_blstm.py"

    # Load vocabularies for both words and punctuations
    word_vocab = load_vocab_to_dict(os.path.join(vocab_dir, "words.txt"))
    punc_vocab = load_vocab_to_dict(os.path.join(vocab_dir, "puncs.txt"))

    # Load dataset / vocabulary params
    params = load_json_to_dict(os.path.join(vocab_dir, "params.json"))

    return word_vocab, punc_vocab, params


def generate_punctuations(cfg, in_file, vocab_dir, model_checkpoint, out_file):
    """Generate punctuations for text using trained BLSTM token classification model"""

    # Load text to be punctuated
    text_instances = load_text_instances(in_file)

    # Load model vocabularies
    word_vocab, punc_vocab, vocab_params = load_vocabs(vocab_dir)
    inv_punc_vocab = {idx: punc for punc, idx in punc_vocab.items()}

    # Specify the device to be used for the generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = PhraseBreakPredictor(
        num_words=len(word_vocab),
        word_embedding_dim=cfg["word_embedding_dim"],
        num_blstm_layers=cfg["num_blstm_layers"],
        blstm_layer_size=cfg["blstm_layer_size"],
        num_puncs=len(punc_vocab),
        padding_idx=word_vocab[vocab_params["words_pad_token"]],
    )
    model = model.to(device)

    # Load the trained model from checkpoint
    model = load_checkpoint_to_evaluate_model(model_checkpoint, model, device)

    model.eval()
    with torch.no_grad():
        for fileid, unpunc_paragraph in text_instances:
            print(f"Processing {fileid}")

            # Split the paragraph into constituent sentences
            unpunc_paragraph = [sentence for sentence in unpunc_paragraph.split(".") if sentence != ""]

            # punc_paragraph = []

            # Process each sentence in the paragraph
            for unpunc_text in unpunc_paragraph:
                print(unpunc_text)
                # Convert the unpunctuated text to sequence of word ids; for prediction by the model
                unpunc_text = [word for word in unpunc_text.split()]
                unpunc_text_seq = torch.LongTensor(
                    [
                        word_vocab[word] if word in word_vocab else word_vocab[vocab_params["words_unk_token"]]
                        for word in unpunc_text
                    ]
                ).unsqueeze(0)
                unpunc_text_seq = unpunc_text_seq.to(device)

                # Predict sequence of punctuation probabilities using the trained model
                pred_probs_seq = model(unpunc_text_seq)
                pred_probs_seq = pred_probs_seq.view(-1, pred_probs_seq.shape[2]).contiguous()

                # Find punctuation predicted for each word by the model
                _, pred_puncs_seq = torch.max(pred_probs_seq, 1)
                pred_puncs = list(pred_puncs_seq.cpu().numpy())
                pred_puncs = [inv_punc_vocab[cid] for cid in pred_puncs]

                punc_text = [token for word_punc_pair in zip(unpunc_text, pred_puncs) for token in word_punc_pair]
                punc_text = " ".join(punc_text)
                print(punc_text)
                # punc_text = punc_text.replace(" _NONE_ ", " ")
                # punc_text = punc_text.replace(" _COMMA_ ", ", ")
                # punc_text = punc_text.replace(" _PERIOD_ ", ".")

                # punc_paragraph.append(punc_text)

            # print(" ".join(punc_paragraph))


if __name__ == "__main__":
    # Setup parser to parse and accept command line arguments
    parser = argparse.ArgumentParser(
        description="Generate text with punctuations using trained BLSTM token classification model"
    )

    parser.add_argument("--config_file", help="Path to file containing model configuration to be loaded", required=True)
    parser.add_argument("--in_text_file", help="Path to text file containing unpunctuated text", required=True)
    parser.add_argument("--vocab_dir", help="Directory containing vocab files used to train the model", required=True)
    parser.add_argument(
        "--model_checkpoint",
        help="Path to the checkpoint containing the trained model to be used for generation",
        required=True,
    )
    parser.add_argument("--out_text_file", help="Output file where punctuated text will be written", required=True)

    # Parse and get command line arguments
    args = parser.parse_args()

    # Load configuration from file
    assert os.path.isfile(args.config_file), f"No config file found at {args.config_file}"
    cfg = load_json_to_dict(args.config_file)

    # Generate text with punctuations using the trained model
    generate_punctuations(cfg, args.in_text_file, args.vocab_dir, args.model_checkpoint, args.out_text_file)
