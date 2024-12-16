import os
import argparse
from training import train_dataset_partitions

# Example Usage:
# This will train a series of models on the specified dataset using the dataset partition and augmentation configurations specified in the json file.
# The dataset map generated when training the baseline model is required.
# python train_models_with_dataset_partitions.py data/partitions_configurations.json data/dataset_map.csv --model_id google/electra-small-discriminator --train_dataset_id squad --ood_dataset_id UCLNLP/adversarial_qa --num_epochs 1 --batch_size 1 --output_dir ./output/

MODEL_ID = os.environ.get("MODEL_ID", "google/electra-small-discriminator")
TRAIN_DATASET_ID = os.environ.get("TRAIN_DATASET_ID", "squad")
OOD_DATASET_ID = os.environ.get("OOD_DATASET_ID", "UCLNLP/adversarial_qa")
NUM_EPOCHS = int(os.environ.get("PARTITIONS_NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("PARTITIONS_BATCH_SIZE", "5"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", '/opt/ml/output/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate augmentations for a dataset')
    parser.add_argument('partitions_configuration_path', type=str, help='Path to the json file containing the partitions configurations')
    parser.add_argument('dataset_map_path', type=str, help='Path to the csv file containing the dataset map generated when training the baseline model')
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Id or path to the model to train')
    parser.add_argument('--train_dataset_id', type=str, default=TRAIN_DATASET_ID, help='Id or path to the dataset to be used for training and in-domain validation')
    parser.add_argument('--ood_dataset_id', type=str, default=OOD_DATASET_ID, help='Id or path to the dataset to be used for out-of-domain validation')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save the results including model checkpoints and validation results')
    args = parser.parse_args()

    train_dataset_partitions(
        partitions_configuration_path=args.partitions_configuration_path,
        dataset_map_path=args.dataset_map_path,
        model_id=args.model_id,
        train_dataset_id=args.train_dataset_id,
        ood_dataset_id=args.ood_dataset_id,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
        
    