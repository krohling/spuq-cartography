import time
import torch
import tqdm
import datasets
import torch.nn.functional as F
import pandas as pd
import bert_score
from rouge_score import rouge_scorer
from util import process_datasets, postprocess_qa_predictions
from .compute_score import f1_score, exact_match_score

import torch
import torch.nn.functional as F

def compute_intersample_confidence(target, perturbations, text_scorer_fn):
    target_context, target_span = target["context"], target["pred_span"]
    conf_score, conf_norm = 0.0, 0.0

    for _, perturbation in perturbations.iterrows():
        weight = text_scorer_fn(target_context, perturbation['context'])
        conf_norm += weight

        similarity = text_scorer_fn(target_span, perturbation["pred_span"])
        conf_score += similarity*weight

    return conf_score/conf_norm


def compute_cartography_metrics(trainer, augmentation_dataset_id, output_path, text_scorer='rougeL', keep_original_only=False):
    start_time = time.time()
    print(f"Trainer device: {trainer.model.device}")
    print("Processing datasets...")
    examples_dataset = datasets.load_dataset(augmentation_dataset_id)
    examples_dataset['validation'] = examples_dataset['train']
    _, perturbations_tokenized = process_datasets(examples_dataset, trainer.tokenizer)
    examples_dataset = examples_dataset['train']
    print(f"Processing datasets took {time.time() - start_time:.2f} seconds")

    if text_scorer.lower() == 'bertscore':
        print("Using BERTScore for text similarity")
        scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
        text_scorer_fn = lambda x, y: max(scorer.score([x], [y])[2].item(), 0.0)
    else:
        print("Using ROUGE-L for text similarity")
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        text_scorer_fn = lambda x, y: scorer.score(x, y)['rougeL'].fmeasure
    

    print("Computing predictions...")
    start_time = time.time()
    with torch.no_grad():
        output = trainer.predict(perturbations_tokenized)
    eval_preds = postprocess_qa_predictions(examples_dataset, perturbations_tokenized, output.predictions)
    print(f"Computing predictions took {time.time() - start_time:.2f} seconds")

    
    examples_table = {row['id']: row for row in examples_dataset.to_list()}
    example_ids = perturbations_tokenized['example_id']


    print("Computing intrasample metrics table...")
    start_time = time.time()
    metrics = []
    for i in tqdm.tqdm(range(output.predictions[0].shape[0])):
        example_id = example_ids[i]
        example = examples_table[example_id]

        label_answer = example['answers']['text'][0]
        pred_start_probs = F.softmax(torch.tensor(output.predictions[0][i]), dim=-1).squeeze()
        pred_end_probs = F.softmax(torch.tensor(output.predictions[1][i]), dim=-1).squeeze()
        pred_start_position = torch.argmax(pred_start_probs).item()
        pred_end_position = torch.argmax(pred_end_probs).item()

        pred_start_confidence = pred_start_probs[pred_start_position].item()
        pred_end_confidence = pred_end_probs[pred_end_position].item()
        pred_confidence = (pred_start_confidence + pred_end_confidence) / 2

        pred_span = eval_preds[example_id]

        em = exact_match_score(pred_span, label_answer)
        f1 = f1_score(pred_span, label_answer)

        metrics.append({**example,
            "pred_span": pred_span,
            "intrasample_confidence": pred_confidence,
            "em": em,
            "f1": f1
        })
    print(f"Computing intrasample metrics took {time.time() - start_time:.2f} seconds")


    print("Computing intersample metrics...")
    start_time = time.time()
    df_metrics = pd.DataFrame(metrics)
    squad_groups = df_metrics.groupby('squad_id')
    metrics = []
    
    for squad_id, examples in tqdm.tqdm(squad_groups):
        if len(examples) > 1:
            target = examples[examples['id'] == squad_id].iloc[0]
            perturbations = examples[examples['id'] != squad_id]
            intersample_confidence = compute_intersample_confidence(target, perturbations, text_scorer_fn)
            metrics.append({
                **target,
                "intersample_confidence": intersample_confidence,
                "agg_confidence": (target["intrasample_confidence"] + intersample_confidence) / 2
            })
        else:
            print(f"Skipping example {squad_id} with no perturbations")
    
    print(f"Computing intersample metrics took {time.time() - start_time:.2f} seconds")

    df_results = pd.DataFrame(metrics)
    if keep_original_only:
        df_results = df_results[df_results["id"] == df_results["squad_id"]]
    df_results.drop(columns=["squad_id", "title","context", "question", "answers", "pred_span"], inplace=True)
    df_results.to_csv(output_path, index=False)
