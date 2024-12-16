python train_baseline_model.py --output_dir "./output/baseline"
python train_models_with_dataset_partitions.py ./data/partitions_config.json ./output/baseline/final_dataset_map.csv --output_dir ./output/partitions/
echo "***Training Complete***"
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "Terminating Pod"
    runpodctl remove pod $RUNPOD_POD_ID
fi