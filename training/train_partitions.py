import os
import json
import shutil
import numpy as np
import wandb
import datasets
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments
)
from util import process_datasets, QuestionAnsweringTrainer
from training.callbacks import ValidationTrainerCallback

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", None)
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", None)
WANDB_ENABLED = os.environ.get("WANDB_API_KEY", None) is not None

def train_with_dataset_partition(
        run_name, 
        augmentation_datasets, 
        dataset_config, 
        dataset_map_path, 
        model_id,
        train_dataset_id,
        ood_dataset_id,
        num_epochs,
        batch_size,
        output_dir
    ):

    print("******************************")
    print("Training Model With Dataset Partition")
    print(f"Training model: {model_id}")
    print(f"Training dataset: {train_dataset_id}")
    print(f"Out-of-domain validation dataset: {ood_dataset_id}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print(f"Run name: {run_name}")
    print(f"Augmentation datasets: {augmentation_datasets}")
    print(f"Dataset configuration: {dataset_config}")
    print("******************************")

    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)


    print("******************************")
    print(f"Starting training run:")
    print(run_name)
    print("******************************")

    print("******************************")
    print("Loading model and tokenizer...")
    print("******************************")
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    print("******************************")
    print("Processing datasets...")
    print("******************************")
    dataset = datasets.load_dataset(train_dataset_id)

    dataset_ids = []
    if len(dataset_config) > 0:
        dataset_map = pd.read_csv(dataset_map_path)
        for item in dataset_config:
            count = int(len(dataset_map) * item['percentage'])
            if item['metric'] == 'random':
                dataset_ids.extend(dataset_map.sample(n=count)['id'].tolist())
            else:
                df_filtered_ids = dataset_map.sort_values(
                    by=item['metric'], 
                    ascending=item['sort'] != 'high'
                ).head(count)['id'].tolist()
                dataset_ids.extend(df_filtered_ids)
        
        dataset_ids = list(set(dataset_ids))
        # dataset['train'] = dataset['train'].filter(lambda example: example['id'] in dataset_ids)

    if len(augmentation_datasets) > 0:
        def process_augmenation_dataset(dataset_id):
            paraphrasing_dataset = datasets.load_dataset(dataset_id)
            def filter_augmentation_rows(dataset):
                seen_ids = set()
                def unique_filter(example):
                    squad_id = example['squad_id']
                    if squad_id in seen_ids or example['id'] == squad_id or (squad_id not in dataset_ids and len(dataset_ids) > 0):
                        return False
                    else:
                        seen_ids.add(squad_id)
                        return True
                
                return dataset.filter(unique_filter)
            
            return filter_augmentation_rows(paraphrasing_dataset['train']).remove_columns(["squad_id"])

        def align_answers_format(example):
            example['answers'] = [
                {'text': [text], 'answer_start': [np.int32(answer_start)]}
                for text, answer_start in zip(example['answers']['text'], example['answers']['answer_start'])
            ]
            return example

        formatted_augmentation_datasets = []
        for augmentation_dataset in augmentation_datasets:
            print(f"Augmenting dataset with {augmentation_dataset}...")
            formatted_ds = process_augmenation_dataset(augmentation_dataset)
            formatted_ds = formatted_ds.map(align_answers_format)
            formatted_augmentation_datasets.append(formatted_ds)
            print(f"Dataset size: {len(formatted_ds)}")

        dataset['train'] = dataset['train'].map(align_answers_format)
        formatted_augmentation_datasets.append(dataset['train'])
        dataset['train'] = datasets.concatenate_datasets(formatted_augmentation_datasets)

        def fix_answers_column(example):
            example['answers'] = example['answers'][0]
            return example
        dataset['train'] = dataset['train'].map(fix_answers_column)

        print(f"Augmented dataset size: {len(dataset['train'])}")

    train_dataset, eval_dataset = process_datasets(dataset, tokenizer)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Training dataset configuration: {dataset_config}")

    ood_dataset = datasets.load_dataset(ood_dataset_id, 'adversarialQA')
    _, ood_eval_dataset = process_datasets(ood_dataset, tokenizer)

    

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
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name)
        training_args.report_to = ["wandb"]
        wandb.config.update(training_args)


    trainer = QuestionAnsweringTrainer(
        args=training_args, 
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
    )

    id_validation_output_path = f"{output_dir}/id_validation_results.csv"
    id_validation_callback = ValidationTrainerCallback(trainer, train_dataset_id, eval_dataset, dataset['validation'], id_validation_output_path)
    trainer.add_callback(id_validation_callback)

    ood_validation_output_path = f"{output_dir}/ood_validation_results.csv"
    ood_validation_callback = ValidationTrainerCallback(trainer, 'ood', ood_eval_dataset, ood_dataset['validation'], ood_validation_output_path)
    trainer.add_callback(ood_validation_callback)

    print("******************************")
    print("Starting training...")
    print("******************************")
    trainer.train()
    model_path = f"{output_dir}/final_model"
    trainer.save_model(model_path)
    shutil.make_archive(model_path, 'zip', output_dir)

    if WANDB_ENABLED:
        wandb.save(f"{output_dir}/final_model.zip")
        wandb.save(id_validation_output_path)
        wandb.save(ood_validation_output_path)
        wandb.finish()


def train_dataset_partitions(
        partitions_configuration_path,
        dataset_map_path, 
        model_id,
        train_dataset_id,
        ood_dataset_id,
        num_epochs,
        batch_size,
        output_dir
    ):
    run_configurations = json.load(open(partitions_configuration_path))
    for config in run_configurations:
        train_with_dataset_partition(
            run_name=config['run_name'],
            augmentation_datasets=config['augmentation_datasets'],
            dataset_config=config["dataset_config"],
            dataset_map_path=dataset_map_path,
            model_id=model_id,
            train_dataset_id=train_dataset_id,
            ood_dataset_id=ood_dataset_id,
            num_epochs=num_epochs,
            batch_size=batch_size,
            output_dir=f"{output_dir}/{config['run_name']}"
        )
