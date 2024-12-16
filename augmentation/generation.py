import os
import time
import csv
import numpy as np
import datasets
import uuid
import csv
import uuid
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI

# Prices for gpt-4o-mini-2024-07-18 as of 2024-12-01
INPUT_TOKEN_PRICE = 0.15/1000000
OUTPUT_TOKEN_PRICE = 0.6/1000000

AUGMENTATION_TYPE_PARAPHRASING = 'paraphrasing'
AUGMENTATION_TYPE_ADVERSARIAL = 'adversarial'
AUGMENTATION_TYPE_CONTRAST = 'contrast'
AUGMENTATION_TYPE_ORIGINAL = 'original'

OUTPUT_COLUMNS = ['id', 'squad_id', 'augmentation_type', 'title', 'context', 'question', 'answers']

class AugmentationItem(BaseModel):
    question: str
    context: str

class DataAugmentationResponse(BaseModel):
    augmentations: list[AugmentationItem]


def generate_example_augmentation(system_prompt: str, question: str, context: str, answer: str) -> str:
    client = OpenAI()
    user_prompt = f"""QUESTION: {question}
    
CONTEXT: {context}

ANSWER: {answer}

VERY IMPORTANT: Ensure that the new context contains the ANSWER ('{answer}'). If the answer is not present in the new context, the augmentation will be invalid."""
    completion = client.beta.chat.completions.parse(
        # model="gpt-4o-2024-08-06",
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=DataAugmentationResponse,
    )

    cost = completion.usage.prompt_tokens * INPUT_TOKEN_PRICE + completion.usage.completion_tokens * OUTPUT_TOKEN_PRICE
    new_examples = completion.choices[0].message.parsed.augmentations

    results = []
    for new_example in new_examples:
        if answer.lower() in new_example.context.lower():
            answer_start = new_example.context.lower().index(answer.lower())
            results.append((new_example.question, new_example.context, answer_start))

    return cost, results



def generate_paraphrased_examples(question: str, context: str, answer: str, count: int) -> str:
    return generate_example_augmentation(
        f"""You are being provided with QUESTION, CONTEXT, and ANSWER texts.
Your task is to provide {count} paraphrasings of both the QUESTION and CONTEXT.
The paraphrasings should maintain the same meaning as the original texts and MUST contain the ANSWER span. 
If the paraphrased CONTEXT does not contain the ANSWER, it will be invalid. The paraphrased question should be semantically similar to the original but can be rephrased for variety.
Here are a few examples to follow:

Example 1:
QUESTION: 'Who was the first president of the United States?'
CONTEXT: 'George Washington was the first president of the United States.'
ANSWER: 'George Washington'
Paraphrased Question: 'Who became the inaugural president of the United States?'
Paraphrased Context: 'The United States' first president was George Washington.'

Example 2:
QUESTION: 'When did the Titanic sink?'
CONTEXT: 'The Titanic sank on April 15, 1912, after hitting an iceberg.'
ANSWER: 'April 15, 1912'
Paraphrased Question: 'What year did the Titanic sink?
Paraphrased Context: 'After striking an iceberg, the Titanic sank on the night of April 15, 1912.'""",
        question, 
        context, 
        answer, 
    )

def generate_adversarial_examples(question: str, context: str, answer: str, count: int) -> str:
    return generate_example_augmentation(
        f"""You are being provided with QUESTION, CONTEXT, and ANSWER texts.
Your task is to provide {count} adversarial versions of the CONTEXT. 
Adversarial versions should add distractors or misleading information that could confuse a model, while keeping the correct ANSWER in the CONTEXT.
Ensure the ANSWER remains accurate in the modified CONTEXT.
Here are a few examples to follow:

Example 1:
QUESTION: 'Where was Albert Einstein born?'
CONTEXT: 'Albert Einstein was born in Ulm, Germany, in 1879.'
ANSWER: 'Ulm, Germany'
Adversarial Context: 'Albert Einstein was born in Ulm, Germany, in 1879. Some believe he was born in Munich, but that is incorrect.'

Example 2:
QUESTION: 'What is the capital of France?'
CONTEXT: 'Paris is the capital of France.'
ANSWER: 'Paris'
Adversarial Context: 'Paris is the capital of France, although some mistakenly think Lyon holds that title.'""",
        question, 
        context, 
        answer
    )

def generate_contrast_examples(question: str, context: str, answer: str, count: int) -> str:
    return generate_example_augmentation(
        f"""You are being provided with QUESTION, CONTEXT, and ANSWER texts.
Your task is to provide {count} contrast versions of both the QUESTION and CONTEXT.
The contrast versions should rephrase or modify the QUESTION and CONTEXT in a way that makes the task harder but still requires the same ANSWER.
Here are a few examples to follow:

Example 1:
QUESTION: 'What is the capital of France?'
CONTEXT: 'Paris is the capital of France.'
ANSWER: 'Paris'
Contrast Question: 'Which city serves as the capital of France?'
Contrast Context: 'The city of Paris holds the title of France's capital.'

Example 2:
QUESTION: 'Where was Albert Einstein born?'
CONTEXT: 'Albert Einstein was born in Ulm, Germany, in 1879.'
ANSWER: 'Ulm, Germany'
Contrast Question: 'In which city was the physicist Albert Einstein born?'
Contrast Context: 'In 1879, a city in Germany called Ulm became the birthplace of physicist Albert Einstein.'""",
        question, 
        context, 
        answer
    )


def process_example(source_example, augmentations_df, augmentation_counts, include_original):
    results, acc_cost = [], 0
    example_id = source_example['id']
    title = source_example['title']
    question = source_example['question']
    context = source_example['context']
    answer = source_example['answer']
    answer_char_start = source_example['answer_start']

    try:
        all_augmentations = []
        paraphrasing_count, adversarial_count, contrast_count = augmentation_counts
        def generate_augmentations(augmentation_type, augmentation_count, augmentation_fn, max_attempts=3):
            nonlocal acc_cost
            nonlocal all_augmentations
            attempts = 0
            current_count = augmentations_df[
                (augmentations_df['squad_id'] == example_id) & 
                (augmentations_df['augmentation_type'] == augmentation_type)
            ].shape[0]
            target_count = augmentation_count - current_count

            while target_count > 0 and attempts < max_attempts:
                attempts += 1
                try:
                    cost, augmentations = augmentation_fn(
                        question, 
                        context, 
                        answer, 
                        target_count
                    )
                    acc_cost += cost
                    target_count -= len(augmentations)
                    augmentations = [a + (augmentation_type,) for a in augmentations]
                    all_augmentations.extend(augmentations)
                except Exception as e:
                    if attempts >= max_attempts:
                        print(f"Failed to generate {augmentation_type} examples for {example_id}, error: {str(e)}")

        generate_augmentations(AUGMENTATION_TYPE_PARAPHRASING, paraphrasing_count, generate_paraphrased_examples)
        generate_augmentations(AUGMENTATION_TYPE_ADVERSARIAL, adversarial_count, generate_adversarial_examples)
        generate_augmentations(AUGMENTATION_TYPE_CONTRAST, contrast_count, generate_contrast_examples)

        if include_original and augmentations_df[(augmentations_df['id'] == example_id) & (augmentations_df['augmentation_type'] == AUGMENTATION_TYPE_ORIGINAL)].shape[0] == 0:
            results.append({
                'id': example_id,
                'squad_id': example_id,
                'augmentation_type': AUGMENTATION_TYPE_ORIGINAL,
                'title': title,
                'context': context,
                'question': question,
                'answers': {
                    "text": [answer],
                    "answer_start": [answer_char_start],
                },
            })

        for new_question, new_context, answer_char_start, augmentation_type in all_augmentations:
            try:
                results.append({
                    'id': str(uuid.uuid4()),
                    'squad_id': example_id,
                    'augmentation_type': augmentation_type,
                    'title': title,
                    'context': new_context,
                    'question': new_question,
                    'answers': {
                        "text": [answer],
                        "answer_start": [answer_char_start],
                    }
                })
            except Exception as e:
                print(f"Skipping augmentation_type, error: {str(e)}")
    except Exception as e:
        print(f"Skipping example {example_id}, error: {str(e)}")

    return acc_cost, results

def process_examples(
        examples, 
        augmentations_df, 
        augmentation_counts, 
        output_filename, 
        include_original, 
        acc_cost: float=0,
        batch_size=25, 
        num_workers=4,
        worker_timeout=60
    ):
    results = []
    if len(examples) > batch_size:
        example_batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
    else:
        example_batches = [examples]

    print(f"Processing {len(examples)} examples in {len(example_batches)} batches...")
    for batch, example_batch in enumerate(example_batches):
        batch_results, batch_futures = [], []
        batch_start = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i, example in enumerate(example_batch):
                future = executor.submit(process_example, example, augmentations_df, augmentation_counts, include_original)
                batch_futures.append(future)

            for future in batch_futures:
                try:
                    cost, results = future.result(timeout=worker_timeout)
                    acc_cost += cost
                    batch_results.extend(results)
                except TimeoutError:
                    print("TimeoutError")
        
        results.extend(batch_results)
        if len(batch_results) > 0:
            with open(output_filename, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
                writer.writerows(batch_results)

        print(f"Completed Batch {batch+1} of {len(example_batches)} in {time.time()-batch_start}s Current Cost: {acc_cost}")
    
    return acc_cost, results

def print_augmentations_details(augmentations_path: str):
    augmentations_df = pd.read_csv(augmentations_path)

    print("**************************************************")
    print(f"Total Number of Examples: {len(augmentations_df)}")
    print(f"Number of Original Examples: {len(augmentations_df[augmentations_df['augmentation_type'] == AUGMENTATION_TYPE_ORIGINAL])}")
    print(f"Number of Paraphrasing Examples: {len(augmentations_df[augmentations_df['augmentation_type'] == AUGMENTATION_TYPE_PARAPHRASING])}")
    print(f"Number of Adversarial Examples: {len(augmentations_df[augmentations_df['augmentation_type'] == AUGMENTATION_TYPE_ADVERSARIAL])}")
    print(f"Number of Contrast Examples: {len(augmentations_df[augmentations_df['augmentation_type'] == AUGMENTATION_TYPE_CONTRAST])}")
    print("**************************************************\n\n")
    



def generate_augmentations(
        output_filename: str, 
        dataset_id: str='squad',
        augmentation_counts=[1., 1., 1.],
        batch_size: int=25,
        num_workers: int=4,
        include_original: bool=False,
        skip_remidiation: bool=True
    ) -> pd.DataFrame:
    
    start_time = time.time()
    print("Loading dataset...")
    dataset = datasets.load_dataset(dataset_id)['train']
    examples = [
        {
            'id': example['id'],
            'title': example['title'],
            'question': example['question'],
            'context': example['context'],
            'answer': example['answers']['text'][0],
            'answer_start': example['answers']['answer_start'][0],
        }
        for example in dataset
    ]
    examples = examples[:50]

    print("**************************************************")
    print(f"Dataset size: {len(examples)}")
    print(f"{augmentation_counts[0]} paraphrasing augmentation(s) will be generated for each record.")
    print(f"{augmentation_counts[1]} adversarial augmentation(s) will be generated for each record.")
    print(f"{augmentation_counts[2]} contrast augmentation(s) will be generated for each record.")
    print("**************************************************\n\n")

    if os.path.exists(output_filename):
        print("Loading existing augmentations...")
        print_augmentations_details(output_filename)
        augmentations_df = pd.read_csv(output_filename)
    else:
        print("No existing augmentations found.")
        augmentations_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        augmentations_df.to_csv(output_filename, index=False)
    

    all_results, total_cost = [], 0
    if not skip_remidiation:
        print("Remediating existing augmentations...")
        examples_to_remediate = augmentations_df['squad_id'].unique()
        examples_to_remediate = [example for example in examples if example['id'] not in examples_to_remediate]
        cost, results = process_examples(
            examples=examples_to_remediate, 
            augmentations_df=augmentations_df, 
            augmentation_counts=augmentation_counts, 
            output_filename=output_filename, 
            include_original=include_original, 
            acc_cost=total_cost,
            batch_size=batch_size, 
            num_workers=num_workers
        )
        total_cost += cost
        all_results.extend(results)
        print(f"Completed remediation. Total Cost: {total_cost}")
        print(f"Total Time Elapsed: {time.time() - start_time}\n\n")
        print_augmentations_details(output_filename)

    print("Generating augmentations for new examples...")
    cost, results = process_examples(
        examples=examples, 
        augmentations_df=augmentations_df, 
        augmentation_counts=augmentation_counts, 
        output_filename=output_filename, 
        include_original=include_original, 
        acc_cost=total_cost,
        batch_size=batch_size, 
        num_workers=num_workers
    )
    total_cost += cost
    all_results.extend(results)
    

    print(f"Completed augmentation generation. Total Cost: {total_cost}")
    print(f"Total Time Elapsed: {time.time() - start_time}\n\n")
    print_augmentations_details(output_filename)

