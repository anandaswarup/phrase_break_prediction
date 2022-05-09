"""Preprocess the downloaded LibriTTS Label dataset and save it in a convenient format"""

import argparse
import os
from glob import glob


def process_lab_file(filename):
    """Process a single lab file
    """
    words, tags = [], []

    with open(filename, "r") as file_reader:
        lab = file_reader.readlines()

    lab = [line.strip("\n") for line in lab]
    lab = [line.split("\t") for line in lab]

    for idx in range(len(lab) - 1):
        word = lab[idx][2]
        if word == "":
            continue
        else:
            next_word = lab[idx + 1][2]
            tag = "B" if next_word == "" else "NB"

            words.append(word)
            tags.append(tag)

    tags[-1] = "BB"

    return words, tags


def export_dataset(dataset, output_dir):
    """Export a processed dataset to disk
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "sentences.txt"), "w") as sentence_writer:
        with open(os.path.join(output_dir, "labels.txt"), "w") as label_writer:
            for words, tags in dataset:
                sentence_writer.write("{}\n".format(" ".join(words)))
                label_writer.write("{}\n".format(" ".join(tags)))


def process_LibriTTS_label_dataset(dataset_dir, out_dir):
    """Process the raw dataset
    """
    # Get filenames for train/dev/test splits
    train_filenames = glob(f"{dataset_dir}/lab/word/train-clean-360/**/*.lab", recursive=True)
    dev_filenames = glob(f"{dataset_dir}/lab/word/dev-clean/**/*.lab", recursive=True)
    test_filenames = glob(f"{dataset_dir}/lab/word/test-clean/**/*.lab", recursive=True)

    # Process the datasets
    train_dataset, dev_dataset, test_dataset = [], [], []

    print("Processing train split")
    for train_filename in train_filenames:
        words, tags = process_lab_file(train_filename)
        train_dataset.append((words, tags))

    print("Processing dev split")
    for dev_filename in dev_filenames:
        words, tags = process_lab_file(dev_filename)
        dev_dataset.append((words, tags))

    print("Processing test split")
    for test_filename in test_filenames:
        words, tags = process_lab_file(test_filename)
        test_dataset.append((words, tags))

    # Export the datasets
    print("Exporting train dataset")
    export_dataset(train_dataset, os.path.join(out_dir, "train"))
    print("Exporting dev dataset")
    export_dataset(dev_dataset, os.path.join(out_dir, "dev"))
    print("Exporting test dataset")
    export_dataset(test_dataset, os.path.join(out_dir, "test"))


if __name__ == "__main__":
    # setup parser to accept command line arguments
    parser = argparse.ArgumentParser(
        description="Preprocess the downloaded LibriTTS label dataset and save it in a convenient format"
    )

    # Command line arguments
    parser.add_argument("--raw_dataset_dir", help="Path to the downloaded dataset", required=True)
    parser.add_argument(
        "--processed_dataset_dir", help="Output dir, where the processed dataset will be written", required=True
    )

    # Parse and get the command line arguments
    args = parser.parse_args()
    raw_dataset_dir = args.raw_dataset_dir
    processed_dataset_dir = args.processed_dataset_dir

    # Process the raw dataset
    process_LibriTTS_label_dataset(raw_dataset_dir, processed_dataset_dir)
