"""Preprocess the downloaded LibriTTS Label dataset and transform to a format suitable for the model"""

import argparse
import os
from glob import glob


def process_lab_file(labfile):
    """Process lab file
    """
    words, puncs = [], []

    with open(labfile, "r") as file_reader:
        lab = file_reader.readlines()

    lab = [line.strip("\n") for line in lab]
    lab = [line.split("\t") for line in lab]

    for idx in range(len(lab) - 1):
        word = lab[idx][2]
        if word == "":
            continue
        else:
            next_word = lab[idx + 1][2]
            punc = "_COMMA_" if next_word == "" else "_NONE_"

            words.append(word)
            puncs.append(punc)

    puncs[-1] = "_PERIOD_"

    return words, puncs


def export_dataset(dataset_split, output_dir):
    """Export a processed dataset split to disk
        
        Args:
            dataset (list of tuples): dataset to be written to disk
            output_dir (str): path to the dir where the dataset will be written
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "sentences.txt"), "w") as sentence_writer:
        with open(os.path.join(output_dir, "puncs.txt"), "w") as punc_writer:
            for words, puncs in dataset_split:
                sentence_writer.write("{}\n".format(" ".join(words)))
                punc_writer.write("{}\n".format(" ".join(puncs)))


def process_LibriTTS_label_dataset(dataset_dir, output_dir):
    """Process the raw dataset

        Args:
            dataset_dir (str): path to the raw dataset
            out_dir (str): path to the dir where the processed dataset will be written
    """
    # Get lists of filenames for train/dev/test splits
    train_filenames = glob(f"{dataset_dir}/lab/word/train-clean-360/**/*.lab", recursive=True)
    dev_filenames = glob(f"{dataset_dir}/lab/word/dev-clean/**/*.lab", recursive=True)
    test_filenames = glob(f"{dataset_dir}/lab/word/test-clean/**/*.lab", recursive=True)

    # Process the splits
    train_split, dev_split, test_split = [], [], []

    print("Processing train split")
    for train_filename in train_filenames:
        words, puncts = process_lab_file(train_filename)
        train_split.append((words, puncts))

    print("Processing dev split")
    for dev_filename in dev_filenames:
        words, puncts = process_lab_file(dev_filename)
        dev_split.append((words, puncts))

    print("Processing test split")
    for test_filename in test_filenames:
        words, puncts = process_lab_file(test_filename)
        test_split.append((words, puncts))

    # Export the dataset splits to disk
    print("Exporting train dataset")
    export_dataset(train_split, os.path.join(output_dir, "train"))
    print("Exporting dev dataset")
    export_dataset(dev_split, os.path.join(output_dir, "dev"))
    print("Exporting test dataset")
    export_dataset(test_split, os.path.join(output_dir, "test"))


if __name__ == "__main__":
    # setup parser to accept command line arguments
    parser = argparse.ArgumentParser(
        description="Preprocess the downloaded LibriTTS Label dataset and transform to a format suitable for the model"
    )

    # Command line arguments
    parser.add_argument("--dataset_dir", help="Path to the downloaded dataset", required=True)
    parser.add_argument("--output_dir", help="Output dir, where the transformed dataset will be written", required=True)

    # Parse and get the command line arguments
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # Process the raw dataset
    process_LibriTTS_label_dataset(dataset_dir, output_dir)
