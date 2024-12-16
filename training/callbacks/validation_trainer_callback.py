import evaluate
import csv
import wandb
import torch
from transformers import TrainerCallback
from util import postprocess_qa_predictions

class ValidationTrainerCallback(TrainerCallback):

    def __init__(self, trainer, dataset_name, eval_dataset, eval_examples, output_path):
        self._dataset_name = dataset_name
        self._eval_dataset = eval_dataset
        self._eval_examples = eval_examples
        self._output_path = output_path
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        print("******************************")
        print(f"Computing validation metrics for {self._dataset_name}...")
        print("******************************")

        with torch.no_grad():
            output = self._trainer.predict(self._eval_dataset)
        eval_preds = postprocess_qa_predictions(self._eval_examples,
                                                    self._eval_dataset,
                                                    output.predictions)
        formatted_predictions = [{"id": k, "prediction_text": v}
                                    for k, v in eval_preds.items()]
        references = [{"id": ex["id"], "answers": ex['answers']}
                        for ex in self._eval_examples]

        epoch = int(state.epoch)
        metric = evaluate.load('squad')
        results = metric.compute(
            predictions=formatted_predictions, references=references
        )

        print("******************************")
        print(results)
        print("******************************")
        wandb.log({
            "val_epoch": epoch,
            f"val_{self._dataset_name}_em": results['exact_match'],
            f"val_{self._dataset_name}_f1": results['f1'],
        })

        f_mode = 'w' if epoch == 1 else 'a'
        with open(self._output_path, f_mode, newline='') as file:
            writer = csv.writer(file)
            if f_mode == 'w':
                writer.writerow(['epoch', 'em', 'f1'])
            writer.writerow([epoch, results['exact_match'], results['f1']])

        return control
