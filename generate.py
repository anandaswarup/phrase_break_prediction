"""Generate text with phrase breaks for synthesis"""

import argparse
import os

import numpy as np
import torch

from model import PhraseBreakPredictor
from utils import Config


def load_synthesis_instances(synthesis_file):
    """Read text to be synthesized from file
    """
    with open(synthesis_file, "r") as file_reader:
        synthesis_instances = file_reader.readlines()

    synthesis_instances = [instance.strip("\n") for instance in synthesis_instances]

    synthesis_instances = [instance.split("|") for instance in synthesis_instances]

    return synthesis_instances


def load_trained_model(model_checkpoint, model, device):
    """Load checkpoint from specified path
    """
    checkpoint = torch.load(model_checkpoint, map_location=device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


def load_vocabs(vocab_dir):
    """Load vocab files
    """
    # Loading dataset params
    dataset_params_path = os.path.join(vocab_dir, "dataset_params.json")
    assert os.path.isfile(dataset_params_path), f"No json file found at {dataset_params_path}, run build_vocab.py"
    dataset_params = Config(dataset_params_path)

    # Loading vocabulary (to map words to indices)
    vocab_path = os.path.join(vocab_dir, "words.txt")
    vocab_map = {}
    with open(vocab_path, "r") as reader:
        for idx, word in enumerate(reader.read().splitlines()):
            vocab_map[word] = idx

    # Loading tags (to map tags to indices)
    tags_path = os.path.join(vocab_dir, "tags.txt")
    tag_map = {}
    with open(tags_path, "r") as reader:
        for idx, tag in enumerate(reader.read().splitlines()):
            tag_map[tag] = idx

    return dataset_params, vocab_map, tag_map


def generate(cfg, synthesis_file, vocab_dir, model_checkpoint, out_file):
    """Generate text with phrase breaks for synthesis
    """
    # Load vocabularies
    dataset_params, vocab_map, tag_map = load_vocabs(vocab_dir)
    inv_tag_map = {idx: tag for tag, idx in tag_map.items()}

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = PhraseBreakPredictor(cfg, vocab_size=dataset_params.vocab_size, num_tags=dataset_params.num_tags)
    model = model.to(device)

    # Load the trained model
    model = load_trained_model(model_checkpoint, model, device)

    # Load the synthesis instances
    with open(out_file, "w") as writer:
        synthesis_instances = load_synthesis_instances(synthesis_file)
        for fileid, text in synthesis_instances:
            print(f"Processing {fileid}")

            text = text.replace(",", "")
            text = text.replace(";", "")
            text = text.replace('"', "")
            text = text.replace(":", "")
            text = text.replace("!", ".")
            text = text.replace("?", ".")
            text = text.replace(".", " . ")

            sentences = text.split(".")
            sentences = [sentence.strip() for sentence in sentences if sentence != " "]

            processed_sentences = []
            for sentence in sentences:
                sentence = [word for word in sentence.split(" ")]
                word_seq = torch.LongTensor(
                    [vocab_map[word] if word in vocab_map else vocab_map[dataset_params.unk] for word in sentence]
                ).unsqueeze(0)
                word_seq = word_seq.to(device)

                with torch.no_grad():
                    pred_seq = model(word_seq)

                # np.argmax gives the class predicted for word by tge model
                pred_tag_seq = np.argmax(pred_seq.data.cpu().numpy(), axis=1)
                pred_break_seq = [inv_tag_map[tag] for tag in pred_tag_seq]

                processed_sentence = ""
                for idx in range(len(pred_break_seq)):
                    if pred_break_seq[idx] == "B" and idx < (len(sentence) - 1):
                        processed_sentence += sentence[idx] + "," + " "
                    else:
                        processed_sentence += sentence[idx] + " "

                processed_sentences.append(processed_sentence)
            writer.write(fileid + "|" + ". ".join(processed_sentences) + "\n")


if __name__ == "__main__":
    # Setup parser to parse and accept command line arguments
    parser = argparse.ArgumentParser(description="Generate text with phrase breaks for synthesis")

    parser.add_argument("--synthesis_file", help="Synthesis file containing text to be synthesized", required=True)
    parser.add_argument("--config_file", help="Configuration file (json file)", required=True)
    parser.add_argument("--vocab_dir", help="Directory containing the vocab files", required=True)
    parser.add_argument(
        "--model_checkpoint", help="Trained model checkpoint to use for predicting phrase breaks", required=True
    )
    parser.add_argument(
        "--out_file", help="Output file where generated text with phrase breaks will be written", required=True
    )

    # Parse and get command line arguments
    args = parser.parse_args()

    # Load the configuration parameters from file
    assert os.path.isfile(args.config_file), f"No file found at {args.config_file}"
    cfg = Config(args.config_file)

    generate(cfg, args.synthesis_file, args.vocab_dir, args.model_checkpoint, args.out_file)
