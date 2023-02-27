"""Generate text with punctuations using fine tuned BERT model with token classification head"""
import argparse
import os
import torch
from model.bert import BERTPhraseBreakPredictor
from utils.utils import load_checkpoint_to_evaluate_model, load_json_to_dict, load_vocab_to_dict
from transformers import BertTokenizerFast


def load_text_instances(synthesis_file):
    """Load text to be punctuated from file"""
    with open(synthesis_file, "r") as file_reader:
        text_instances = file_reader.readlines()

    text_instances = [instance.strip("\n") for instance in text_instances]
    text_instances = [instance.split("|") for instance in text_instances]

    return text_instances


def instantiate_tokenizer(cfg, vocab_dir):
    """Instantiate the tokenizer and load the punctuation vocab file"""
    # Instantiate the tokenizer to tokenize the sentences
    tokenizer = BertTokenizerFast.from_pretrained(cfg["bert_model_name"])

    # Load vocabularies for punctuations
    punc_vocab = load_vocab_to_dict(os.path.join(vocab_dir, "puncs.txt"))

    return tokenizer, punc_vocab


def generate_punctuations(cfg, in_file, vocab_dir, model_checkpoint, out_file):
    """Generate punctuations for text using the fine tuned BERT model with a token classification head"""
    # Load the text to be punctuated
    text_instances = load_text_instances(in_file)

    # Instantiate the tokenizer and the punc vocab
    tokenizer, punc_vocab = instantiate_tokenizer(cfg, vocab_dir)
    inv_punc_vocab = {idx: punc for punc, idx in punc_vocab.items()}

    # Specify the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = BERTPhraseBreakPredictor(model_name=cfg["bert_model_name"], num_puncs=len(punc_vocab))
    model = model.to(device)

    # Load the fine tuned model from the checkpoint
    model = load_checkpoint_to_evaluate_model(checkpoint_path=model_checkpoint, model=model, device=device)

    model.eval()

    with torch.no_grad():
        with open(out_file, "w") as file_writer:
            for fileid, unpunc_paragraph in text_instances:
                print(f"Processing {fileid}")

                # Split the paragraph into constituent sentences
                unpunc_paragraph = [sentence.strip() for sentence in unpunc_paragraph.split(".") if sentence != ""]

                punc_paragraph = []

                # Process each sentence in the paragraph
                for unpunc_text in unpunc_paragraph:
                    # Convert the unpunctuated text to sequence of word ids; for prediction by the model
                    unpunc_text = [word for word in unpunc_text.split()]
                    unpunc_text_seq = tokenizer(
                        unpunc_text,
                        is_split_into_words=True,
                        return_offsets_mapping=True,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )

                    ids = unpunc_text_seq["input_ids"].to(device)
                    mask = unpunc_text_seq["attention_mask"].to(device)

                    # Forward pass (predictions)
                    logits = model(text_ids=ids, attention_masks=mask)

                    # Reshape logits
                    logits = logits.view(-1, logits.shape[2]).contiguous()

                    # Convert logits to prediction ids
                    _, pred_punc_ids = torch.max(logits, 1)
                    pred_punc_ids = list(pred_punc_ids.cpu().numpy())

                    # Get word-piece tokens and prediction corresponding to each token
                    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
                    token_predictions = [inv_punc_vocab[idx] for idx in pred_punc_ids]
                    wp_predictions = list(zip(tokens, token_predictions))

                    # Retain the prediction for the first word-piece of a word and discard the predictions for the
                    # other word-pieces
                    predicted_puncs = []
                    for token_pred, mapping in zip(
                        wp_predictions, unpunc_text_seq["offset_mapping"].squeeze().tolist()
                    ):
                        if mapping[0] == 0 and mapping[1] != 0:
                            predicted_puncs.append(token_pred[1])
                        else:
                            continue

                    punc_text = [
                        token for word_punc_pair in zip(unpunc_text, predicted_puncs) for token in word_punc_pair
                    ]
                    punc_text = " ".join(punc_text)

                    punc_text = punc_text.replace(" _NONE_ ", " ")
                    punc_text = punc_text.replace(" _COMMA_ ", ", ")
                    punc_text = punc_text.replace(" _PERIOD_", ".")

                    punc_paragraph.append(punc_text)

                # Generate the paragraph with predicted punctuations
                punc_paragraph = " ".join(punc_paragraph)

                # Write the paragraph with predicted punctuations to file
                file_writer.write(fileid + "|" + punc_paragraph + "\n")


if __name__ == "__main__":
    # Setup parser to parse and accept command line arguments
    parser = argparse.ArgumentParser(
        description="Generate text with punctuations using fine tuned BERT model with token classification head"
    )

    parser.add_argument("--config_file", help="Path to file containing model configurations", required=True)
    parser.add_argument("--in_text_file", help="Path to text file containing unpunctuated text", required=True)
    parser.add_argument(
        "--vocab_dir", help="Directory containing vocab files used to fine tune the model", required=True
    )
    parser.add_argument(
        "--model_checkpoint",
        help="Path to the checkpoint containing the fine tuned model to be used for generation",
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
