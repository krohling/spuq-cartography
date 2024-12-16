import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_data_map import plot_data_map

def generate_dataset_map(cartography_dir, num_epochs):
    example_metrics = {}
    for epoch in range(1, num_epochs + 1):
        cartography_file = f'{cartography_dir}/cartography_epoch_{epoch}.csv'
        df = pd.read_csv(cartography_file)

        for _, row in df.iterrows():
            example_id = row['id']
            if example_id not in example_metrics:
                example_metrics[example_id] = {
                    'intrasample_confidence': [],
                    'intersample_confidence': [],
                    'agg_confidence': [],
                    'em': [],
                    'f1': []
                }
            
            example_metrics[example_id]['intrasample_confidence'].append(row['intrasample_confidence'])
            example_metrics[example_id]['intersample_confidence'].append(row['intersample_confidence'])
            example_metrics[example_id]['agg_confidence'].append(row['agg_confidence'])
            example_metrics[example_id]['em'].append(row['em'])
            example_metrics[example_id]['f1'].append(row['f1'])
    
    dataset_map = []
    for example_id in example_metrics.keys():
        dataset_map.append({
            'id': example_id,
            'intrasample_confidence_mean': np.mean(example_metrics[example_id]['intrasample_confidence']),
            'intrasample_confidence_std': np.std(example_metrics[example_id]['intrasample_confidence']),
            'intersample_confidence_mean': np.mean(example_metrics[example_id]['intersample_confidence']),
            'intersample_confidence_std': np.std(example_metrics[example_id]['intersample_confidence']),
            'agg_confidence_mean': np.mean(example_metrics[example_id]['agg_confidence']),
            'agg_confidence_std': np.std(example_metrics[example_id]['agg_confidence']),
            'em_mean': np.mean(example_metrics[example_id]['em']),
            'em_std': np.std(example_metrics[example_id]['em']),
            'f1_mean': np.mean(example_metrics[example_id]['f1']),
            'f1_std': np.std(example_metrics[example_id]['f1'])
        })

    output_path = f'{cartography_dir}/dataset_map_epoch_{num_epochs}.csv'
    dataset_map_df = pd.DataFrame(dataset_map)
    dataset_map_df.to_csv(output_path, index=False)
    wandb.save(output_path)

    intrasample_cartography_filename = f'{cartography_dir}/dataset_map_intrasample_epoch_{num_epochs}.png'
    plot_data_map(
        dataset_map_df, 
        output_filename=intrasample_cartography_filename, 
        model='electra-small-discriminator', 
        show_hist=True,
        max_instances_to_plot=20000,
        title='Intrasample Confidence',
        hue_metric='em_mean',
        main_metric='intrasample_confidence_std',
        other_metric='intrasample_confidence_mean',
        correctness_metric='em_mean'
    )
    wandb.save(intrasample_cartography_filename)

    intersample_cartography_filename = f'{cartography_dir}/dataset_map_intersample_epoch_{num_epochs}.png'
    plot_data_map(
        dataset_map_df, 
        output_filename=intersample_cartography_filename, 
        model='electra-small-discriminator', 
        show_hist=True,
        max_instances_to_plot=20000,
        title='Intersample Confidence',
        hue_metric='em_mean',
        main_metric='intersample_confidence_std',
        other_metric='intersample_confidence_mean',
        correctness_metric='em_mean'
    )
    wandb.save(intersample_cartography_filename)

def generate_epoch_charts(cartography_dir, num_epochs):
    epochs = list(range(1, num_epochs + 1))
    intrasample_confidence_mean = []
    intrasample_confidence_std = []
    intersample_confidence_mean = []
    intersample_confidence_std = []
    em_mean = []
    em_std = []
    f1_mean = []
    f1_std = []
    for i in epochs:
        cartography_file = f'{cartography_dir}/cartography_epoch_{i}.csv'
        df = pd.read_csv(cartography_file)
        intrasample_confidence_mean.append(df['intrasample_confidence'].mean())
        intrasample_confidence_std.append(df['intrasample_confidence'].std())
        intersample_confidence_mean.append(df['intersample_confidence'].mean())
        intersample_confidence_std.append(df['intersample_confidence'].std())
        em_mean.append(df['em'].mean())
        em_std.append(df['em'].std())
        f1_mean.append(df['f1'].mean())
        f1_std.append(df['f1'].std())

    # Plot all metrics

    # Create a 4x2 subplot grid
    fig, axs = plt.subplots(4, 2, figsize=(20, 20))
    fig.suptitle('Metrics by Epoch', fontsize=20)

    # Plotting Intrasample Confidence Mean and Std
    axs[0, 0].plot(epochs, intrasample_confidence_mean, label='Mean', color='blue')
    axs[0, 0].legend()
    axs[0, 0].set_title('Intrasample Confidence Mean')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Confidence')

    axs[0, 1].plot(epochs, intrasample_confidence_std, label='Std', color='blue', linestyle='dashed')
    axs[0, 1].legend()
    axs[0, 1].set_title('Intrasample Confidence Std')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Confidence')

    # Plotting Intersample Confidence Mean and Std
    axs[1, 0].plot(epochs, intersample_confidence_mean, label='Mean', color='red')
    axs[1, 0].legend()
    axs[1, 0].set_title('Intersample Confidence Mean')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Confidence')

    axs[1, 1].plot(epochs, intersample_confidence_std, label='Std', color='red', linestyle='dashed')
    axs[1, 1].legend()
    axs[1, 1].set_title('Intersample Confidence Std')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Confidence')

    # Plotting EM Mean and Std
    axs[2, 0].plot(epochs, em_mean, label='Mean', color='green')
    axs[2, 0].legend()
    axs[2, 0].set_title('EM Mean')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('EM')

    axs[2, 1].plot(epochs, em_std, label='Std', color='green', linestyle='dashed')
    axs[2, 1].legend()
    axs[2, 1].set_title('EM Std')
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('EM')

    # Plotting F1 Mean and Std
    axs[3, 0].plot(epochs, f1_mean, label='Mean', color='purple')
    axs[3, 0].legend()
    axs[3, 0].set_title('F1 Mean')
    axs[3, 0].set_xlabel('Epoch')
    axs[3, 0].set_ylabel('F1')

    axs[3, 1].plot(epochs, f1_std, label='Std', color='purple', linestyle='dashed')
    axs[3, 1].legend()
    axs[3, 1].set_title('F1 Std')
    axs[3, 1].set_xlabel('Epoch')
    axs[3, 1].set_ylabel('F1')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title

    
