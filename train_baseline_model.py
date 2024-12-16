import os
import argparse
from training import train_model

# Example Usage:
# This will train the specified model on the squad dataset for 1 epoch with a batch size of 1.
# To successfully compute intersample cartography metrics the specified dataset should include data augmentations.
# At the end of each epoch the model will be evaluated on the validation set and the results will be saved to the specified directory.
# Also at the end of each epoch both intersample and intrasample cartography metrics will be computed and saved to the specified directory.
# python train_baseline_model.py --model_id "google/electra-small-discriminator" --train_dataset_id "squad" --augmentation_dataset_id "kevin510/squad_paraphrasing_5" --num_epochs 1 --batch_size 1 --output_dir "./output/"

MODEL_ID = os.environ.get("MODEL_ID", "google/electra-small-discriminator")
TRAIN_DATASET_ID = os.environ.get("TRAIN_DATASET_ID", "squad")
AUGMENTATION_DATASET_ID = os.environ.get("AUGMENTATION_DATASET_ID", "kevin510/squad_paraphrasing_5")
NUM_EPOCHS = int(os.environ.get("BASELINE_NUM_EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("BASELINE_BATCH_SIZE", "1"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", './output')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate augmentations for a dataset')
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Id or path to the model to train')
    parser.add_argument('--train_dataset_id', type=str, default=TRAIN_DATASET_ID, help='Id or path to the dataset to be used for training and validation')
    parser.add_argument('--augmentation_dataset_id', type=str, default=AUGMENTATION_DATASET_ID, help='Id or path to the dataset to be used for computer intersample cartography metrics')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save the model, validation results and cartography metrics')
    args = parser.parse_args()

    train_model(
        model_id=args.model_id,
        train_dataset_id=args.train_dataset_id,
        augmentation_dataset_id=args.augmentation_dataset_id,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
        
    