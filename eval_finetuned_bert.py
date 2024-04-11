"""Evaluate the fine tuned BERT model with a token classification head"""

import argparse
import os

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.bert_data_loader import BERTPhraseBreakDataset
from model.bert import BERTPhraseBreakPredictor
from utils.utils import load_checkpoint_to_evaluate_model, load_json_to_dict


def _remove_predictions_for_padded_tokens(pred_puncs, puncs):
    """Remove predictions corresponding to padded tokens"""
    pred_puncs_without_padded = []
    puncs_without_padded = []

    for pred_punc, punc in zip(pred_puncs, puncs):
        if punc > -100:
            pred_puncs_without_padded.append(pred_punc)
            puncs_without_padded.append(punc)

    return pred_puncs_without_padded, puncs_without_padded


def evaluate_finetuned_model(cfg, dataset_dir, model_checkpoint):
    """Evaluate the fine tuned BERT model"""
    dataset = BERTPhraseBreakDataset(
        model_name=cfg["bert_model_name"], data_dir=dataset_dir, split="test"
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=dataset.pad_collate,
        pin_memory=False,
        drop_last=False,
    )

    # Specify the device to be used for the eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    num_puncs = len(dataset.punc_vocab)
    model = BERTPhraseBreakPredictor(
        model_name=cfg["bert_model_name"], num_puncs=num_puncs
    )
    model = model.to(device)

    # Load the fine tuned model from the checkpoint
    model = load_checkpoint_to_evaluate_model(
        checkpoint_path=model_checkpoint, model=model, device=device
    )

    model.eval()

    with torch.no_grad():
        predictions, ground_truth = [], []
        for batch in loader:
            text_ids, attention_masks, punc_ids = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )

            # Forward pass (predictions)
            logits = model(text_ids=text_ids, attention_masks=attention_masks)

            # Reshape logits and ground truth
            logits = logits.view(-1, logits.shape[2]).contiguous()
            punc_ids = punc_ids.view(-1).contiguous()

            # Convert logits to prediction ids
            _, pred_punc_ids = torch.max(logits, 1)

            # Remove predictions corresponding to paddings
            pred_punc_ids = list(pred_punc_ids.cpu().numpy())
            punc_ids = list(punc_ids.cpu().numpy())
            pred_punc_ids, punc_ids = _remove_predictions_for_padded_tokens(
                pred_punc_ids, punc_ids
            )

            predictions += pred_punc_ids
            ground_truth += punc_ids

    test_set_F1_score = f1_score(ground_truth, predictions, average="micro")

    return test_set_F1_score


if __name__ == "__main__":
    # Setup command line parser to accept command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate the fine tuned BERT model with a token classification head on the held-out test set"
    )

    # Command line arguments
    parser.add_argument(
        "--config_file",
        help="Path to file containing the fine tuning configuration to be loaded",
        required=True,
    )
    parser.add_argument(
        "--dataset_dir",
        help="Directory containing the processed dataset",
        required=True,
    )
    parser.add_argument(
        "--model_checkpoint",
        help="Path to the checkpoint containing the task specific fine tuned model to be used for eval",
        required=True,
    )

    # Parse and get command line arguments
    args = parser.parse_args()

    config_file = args.config_file
    dataset_dir = args.dataset_dir
    model_checkpoint = args.model_checkpoint

    # Load configuration from file
    assert os.path.isfile(
        args.config_file
    ), f"No config file found at {args.config_file}"
    cfg = load_json_to_dict(config_file)

    # Evaluate the model on the test set
    test_set_F1_score = evaluate_finetuned_model(cfg, dataset_dir, model_checkpoint)
    test_set_F1_score = test_set_F1_score * 100

    print(f"F1 Score: {test_set_F1_score:.2f}")
