import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# *************************************************************************************************
# This code is a modified version of the code from the original Dataset Cartography implementation.
# https://github.com/allenai/cartography/blob/main/cartography/selection/train_dy_filtering.py
# *************************************************************************************************

def plot_data_map(dataframe: pd.DataFrame,
                  output_filename: os.path,
                  hue_metric: str = 'correct.',
                  title: str = '',
                  model: str = 'RoBERTa',
                  show_hist: bool = False,
                  max_instances_to_plot = 100000,
                  main_metric = 'variability',
                  other_metric = 'confidence',
                  correctness_metric = 'correctness'):
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')

    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    # dataframe = dataframe.assign(corr_frac = lambda d: d.correctness / d.correctness.max())
    # dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]

    # Keep the numeric values of `corr_frac` for sorting, but display the string labels.
    # dataframe['correct_numeric'] = dataframe['corr_frac']
    # dataframe['correct.'] = [f"{x:.1f}" for x in dataframe['corr_frac']]
    
    # Sort by the numeric values before plotting
    dataframe = dataframe.sort_values(by=correctness_metric, ascending=False)

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(figsize=(14, 10), )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[:, 0])

    # Make the scatterplot.
    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           hue=hue,
                           palette=pal,
                           style=style,
                           s=30)

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda  text, xyc, bbc : ax0.annotate(text,
                                                          xy=xyc,
                                                          xycoords="axes fraction",
                                                          fontsize=15,
                                                          color='black',
                                                          va="center",
                                                          ha="center",
                                                          rotation=350,
                                                           bbox=bb(bbc))
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc='black')
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc='r')
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc='b')


    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc='right')
    else:
        plot.legend(fancybox=True, shadow=True,  ncol=1)
    plot.set_xlabel(main_metric)
    plot.set_ylabel(other_metric)

    if show_hist:
        plot.set_title(f"{title}-{model} Data Map", fontsize=17)

        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        plott0 = dataframe.hist(column=[main_metric], ax=ax1, color='#622a87')
        plott0[0].set_title('')
        plott0[0].set_xlabel(main_metric)
        plott0[0].set_ylabel('density')

        plott1 = dataframe.hist(column=[other_metric], ax=ax2, color='teal')
        plott1[0].set_title('')
        plott1[0].set_xlabel(other_metric)
        plott1[0].set_ylabel('density')

        plott2 = dataframe.hist(column=[correctness_metric], ax=ax3, color='#86bf91')
        plott2[0].set_title('')
        plott2[0].set_xlabel(correctness_metric)
        plott2[0].set_ylabel('density')

    fig.savefig(output_filename, dpi=300)
