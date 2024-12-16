import time
import wandb
from transformers import TrainerCallback
from cartography import compute_cartography_metrics, generate_dataset_map

class CartographyTrainerCallback(TrainerCallback):

    def __init__(self, trainer, augmentation_dataset_id, output_dir):
        self._trainer = trainer
        self._augmentation_dataset_id = augmentation_dataset_id
        self._output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        print("Computing cartography metrics...")
        epoch = int(state.epoch)
        output_path = f'{self._output_dir}/cartography_epoch_{epoch}.csv'
        start_time = time.time()
        compute_cartography_metrics(
            trainer=self._trainer,
            augmentation_dataset_id=self._augmentation_dataset_id,
            output_path=output_path,
            keep_original_only=True
        )
        # generate_epoch_charts(self._output_dir, epoch)
        generate_dataset_map(self._output_dir, epoch)
        wandb.save(output_path)
        print(f"Done computing cartography metrics in {time.time() - start_time} seconds")

        return control
