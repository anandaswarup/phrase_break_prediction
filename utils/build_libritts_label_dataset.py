"""Preprocess the downloaded LibriTTS Label dataset and transform it to a suitable format"""

import argparse
import os
from glob import glob


class LibriTTSLabelDatasetProcessor:
    """Class for processing the LibriTTS Label dataset"""

    def __init__(self, file_list):
        """Instantiate the processor
        Args:
            file_list (str): List of dataset files to be processed
        """
        # Processor attributes
        self.file_list = file_list

        self.processed_src = []
        self.processed_tgt = []

    def _process_lab_file(self, file_name):
        """Process a single lab file
        Args:
            file_name (str): Name of the lab file to be processed
        """
        text, puncs = [], []

        # Open the lab file and read the contents
        with open(file_name, "r") as file_reader:
            lab = file_reader.readlines()

        # Retain only the words from the lab file and discard the timing information
        lab = [line.strip("\n") for line in lab]
        lab = [line.split("\t") for line in lab]
        words = [line[2] for line in lab]

        # Reconstruct the punctuations from spaces in the words
        for idx in range(len(words) - 1):
            word = words[idx]
            if word == "":
                continue
            else:
                next_word = words[idx + 1]
                if next_word == "" and idx != len(words) - 2:
                    punc = "_COMMA_"
                elif next_word == "" and idx == len(words) - 2:
                    punc = "_PERIOD_"
                else:
                    punc = "_NONE_"

                text.append(word)
                puncs.append(punc)

        return " ".join(text), " ".join(puncs)

    def export_processed_dataset(self, output_dir):
        """Export the processed LibriTTS label dataset to disk
        Args:
            output_dir (str): Path to the location where the processed dataset will be written
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the processed src to disk
        with open(os.path.join(output_dir, "src.txt"), "w") as text_writer:
            for line in self.processed_src:
                text_writer.write("{}\n".format(line))

        # Write the processed tgt to disk
        with open(os.path.join(output_dir, "tgt.txt"), "w") as text_writer:
            for line in self.processed_tgt:
                text_writer.write("{}\n".format(line))

    def process_dataset(self):
        """Process the dataset"""
        for filename in self.file_list:
            src, tgt = self._process_lab_file(filename)

            self.processed_src.append(src)
            self.processed_tgt.append(tgt)


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
        dataset_processor.export_processed_dataset(os.path.join(output_dir, split))
