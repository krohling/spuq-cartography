import argparse
from augmentation import generate_augmentations

# Example Usage:
# This will generate 1 of each type of augmentation for each row in the specified (squad) dataset.
# When generating augmentations 5 workers will be used, each processing 25 examples at a time.
# If generating augmentations for a large dataset it's worth experimenting with the number of workers and batch size to improve performance.
# python ./generate_squad_augmentations.py ./artifacts/augmented_dataset.csv --augmentation_counts 1,1,1 --num_workers 5 --batch_size 25

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate augmentations for a dataset')
    parser.add_argument('output_filename', type=str, help='Path to the output CSV file')
    parser.add_argument('--augmentation_counts', type=str, default='1,1,1', help='How many of each augmentation type for each targeted dataset row: Paraphrasing, Adversarial, and Contrast augmentation')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers to use for parallel processing')
    parser.add_argument('--batch_size', type=int, default=25, help='The number of examples each worker processes at a time')
    parser.add_argument('--perform_remediation', action='store_true', help='If this is set to true and the output file already exists, the script will attempt to remediate any existing records that do not have the appropriate number of augmentations')
    parser.add_argument('--include_original', action='store_true', help='If this is set to true, the original example from the source dataset will be included in the output file')
    args = parser.parse_args()
        
    augmentation_counts = [int(x) for x in args.augmentation_counts.split(',')]
    generate_augmentations(
        dataset_id='squad', 
        output_filename=args.output_filename, 
        augmentation_counts=augmentation_counts,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        skip_remidiation=not args.perform_remediation
    )