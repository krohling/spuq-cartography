import os
import shutil
import wandb
import datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments
)
from util import process_datasets, QuestionAnsweringTrainer
from training.callbacks import (
    CartographyTrainerCallback,
    ValidationTrainerCallback
)

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", None)
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
WANDB_ENABLED = os.environ.get("WANDB_API_KEY", None) is not None

def train_model(
        model_id,
        train_dataset_id,
        augmentation_dataset_id,
        num_epochs,
        batch_size,
        output_dir,
    ):

    print("******************************")
    print("Training Model")
    print(f"Training model: {model_id}")
    print(f"Training dataset: {train_dataset_id}")
    print(f"Augmentation dataset: {augmentation_dataset_id}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print("******************************")

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    cartography_dir = os.path.join(output_dir, "cartography")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(cartography_dir, exist_ok=True)

    print("******************************")
    print("Loading model and tokenizer...")
    print("******************************")
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    print("******************************")
    print("Processing dataset...")
    print("******************************")
    dataset = datasets.load_dataset(train_dataset_id)
    train_dataset, eval_dataset = process_datasets(dataset, tokenizer)

    print("**********************************")
    print("Configuring training parameters...")
    print("**********************************")
    training_args = TrainingArguments(
        do_train=True,
        output_dir=checkpoints_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
    )

    if WANDB_ENABLED:
        wandb.login()
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name='baseline')
        training_args.report_to = ["wandb"]
        wandb.config.update(training_args)

    trainer = QuestionAnsweringTrainer(
        args=training_args, 
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )

    cartography_callback = CartographyTrainerCallback(trainer, augmentation_dataset_id, cartography_dir)
    trainer.add_callback(cartography_callback)

    id_validation_output_path = f"{output_dir}/validation_results.csv"
    id_validation_callback = ValidationTrainerCallback(trainer, train_dataset_id, eval_dataset, dataset['validation'], id_validation_output_path)
    trainer.add_callback(id_validation_callback)

    print("******************************")
    print("Starting training...")
    print("******************************")
    trainer.train()

    model_path = f"{output_dir}/final_model"
    trainer.save_model(model_path)
    shutil.make_archive(model_path, 'zip', output_dir)

    final_dataset_map_path = f'{cartography_dir}/dataset_map_epoch_{num_epochs}.csv'
    shutil.copyfile(final_dataset_map_path, f"{output_dir}/final_dataset_map.csv")

    if WANDB_ENABLED:
        wandb.save(f"{output_dir}/final_model.zip")
        wandb.save(id_validation_output_path)
        wandb.save(final_dataset_map_path)
        wandb.finish()

