#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --output=bert_finetuning_log.txt

dataset="LibriTTS"

# Create directories
mkdir -p /scratch/$USER
mkdir -p /scratch/$USER/datasets/phrase_break_prediction
mkdir -p /scratch/$USER/experiments/phrase_break_prediction/$dataset
mkdir -p /scratch/$USER/experiments/phrase_break_prediction/$dataset/bert

# Dataset copy
echo "Copying dataset from /share1/$USER/datasets/phrase_break_prediction to /scratch/$USER/datasets/phrase_break_prediction"
rsync -a $USER@ada:/share1/$USER/datasets/phrase_break_prediction/${dataset}.tar.gz /scratch/$USER/datasets/phrase_break_prediction

# Dataset extraction
echo "Extracting dataset"
tar -zxf /scratch/$USER/datasets/phrase_break_prediction/${dataset}.tar.gz -C /scratch/$USER/datasets/phrase_break_prediction
rm -f /scratch/$USER/datasets/phrase_break_prediction/${dataset}.tar.gz

echo "BERT finetuning"
python finetune_bert.py --config_file config/bert_finetune.json --dataset_dir /scratch/$USER/datasets/phrase_break_prediction/$dataset --experiment_dir /scratch/$USER/experiments/phrase_break_prediction/$dataset/bert

# Copy checkpoints and alignments back to /share1/$USER/experiments/phrase_break_prediction
echo "Copying training/finetuning artifacts from /scratch/$USER/experiments/phrase_break_prediction/$dataset to /share1/$USER/experiments/phrase_break_prediction"
rsync -a /scratch/$USER/experiments/phrase_break_prediction/$dataset $USER@ada:/share1/$USER/experiments/phrase_break_prediction

# Cleanup
rm -rf /scratch/$USER
