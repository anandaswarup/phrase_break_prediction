"""Preprocess the downloaded LibriTTS Label dataset and transform it to a suitable format"""

import argparse
import os
from glob import glob


class LibriTTSLabelDatasetProcessor:
    """Class for processing the LibriTTS Label dataset"""

    def __init__(self, file_list):
        """Instantiate the processor"""
        # Processor attributes
        self.file_list = file_list
        self.processed_dataset = []

    def _process_lab_file(self, file_name):
        """Process lab file"""
        punctuated_text = []

        # Open the lab file and read the contents
        with open(file_name, "r") as file_reader:
            lab = file_reader.readlines()

        # Retain only the words from the lab file and discard the timing information
        lab = [line.strip("\n") for line in lab]
        lab = [line.split("\t") for line in lab]
        words = [line[2] for line in lab]

        # Reconstruct the punctuations from the words in the lab file
        for idx in range(len(words) - 1):
            word = words[idx]
            if word == "":
                continue
            else:
                next_word = words[idx + 1]
                next_word = words[idx + 1]
                if next_word == "" and idx != len(words) - 2:
                    punctuated_text.append(word + ",")
                else:
                    punctuated_text.append(word)

        return " ".join(punctuated_text)

    def process_dataset(self):
        """Process the raw dataset"""
        for filename in self.file_list:
            punctuated_text = self._process_lab_file(filename)
            self.processed_dataset.append(punctuated_text)

    def export_dataset(self, output_dir):
        """Export the processed dataset to disk"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, "sentences.txt"), "w") as sentence_writer:
            for sentence in self.processed_dataset:
                sentence_writer.write("{}\n".format(sentence))


if __name__ == "__main__":
    # Setup parser to accept command line arguments
    parser = argparse.ArgumentParser(
        description="Preprocess the downloaded LibriTTS Label dataset and transform it to a suitable format"
    )

    # Command line arguments
    parser.add_argument("--dataset_dir", help="Path to the downloaded dataset", required=True)
    parser.add_argument("--output_dir", help="Output dir, where the transformed dataset will be written", required=True)

    # Parse and get command line arguments
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # Process each dataset split
    for split in ("train", "dev", "test"):
        print(f"Processing {split} split")
        filelist = (
            glob(f"{dataset_dir}/lab/word/train-clean-360/**/*.lab", recursive=True)
            if split == "train"
            else glob(f"{dataset_dir}/lab/word/{split}-clean/**/*.lab", recursive=True)
        )
        dataset_processor = LibriTTSLabelDatasetProcessor(filelist)
        dataset_processor.process_dataset()
        dataset_processor.export_dataset(os.path.join(output_dir, split))
