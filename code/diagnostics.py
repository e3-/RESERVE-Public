import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==== Constants ====
E3_colors = [[3, 78, 110], [175, 126, 0], [175, 34, 0], [0, 126, 51], [175, 93, 0], [10, 25, 120]]
E3_colors = np.array(E3_colors) / 255  # E3 color pallete


# ==== Active diagnostics functions in use ====

def n_crossings(pred_trainval):
    """

    Computes number of quantile crossings within CV folds for various target percentile pairs

    Args:
        pred_trainval: Dataframe containing quantile estimates for each CV fold and target percentile

    Returns:
        Dataframe containing number of quantile crossings for each pair of target percentiles within each CV fold
            (only for valid pairs of "lower" and "upper" target percentiles)
    """

    columns = pred_trainval.columns # Get columns
    tau_arr = np.sort(np.unique(np.array([c[0] for c in columns])))  # Tau values need to be sorted
    CV_arr = np.sort(np.unique(np.array([c[1] for c in columns])))

    crossings = {} # Define dictionary to store crossings

    for CV in CV_arr:
        # Look for quantile crossings only in sets of predictions from models trained on same CV fold
        crossings[CV] = {}
        for i, t1 in enumerate(tau_arr):
            for j, t2 in enumerate(tau_arr):
                if t1 < t2: # Only evaluate number of quantile crossings on valid lower/upper target percentile pairs
                    crossings[CV][(t1, t2)] = sum(
                        pred_trainval[t1, CV] > pred_trainval[t2, CV])  # Record number of quantile crossings

    df = pd.DataFrame(crossings)
    df.columns.name = 'CV Fold ID'
    df.index.rename(['Lower Quantile', 'Upper Quantile'], inplace=True)
    return df

def get_color_gradient(colors, num_gradient):
    """
    Generate color gradient base on the base colors and the number of gradient you need. Essentially,
    if you want n gradient, than each color's alpha would be i/n+1, i being the index of the gradient.
    :param colors:  base color to apply gradient on. (N,3)
    :param num_gradient: number of gradient required. M
    :return color_gradient: base color extended into color gradients (N,M,3)
    """
    brightness_gradient = np.expand_dims((np.arange(num_gradient) + 1) / (num_gradient + 1), axis=-1)
    color_gradient = 1 - np.expand_dims((1 - colors), axis=1) * brightness_gradient

    return color_gradient


def plot_uncertainty_groupedby_feature(pred_trainval, input_var_discretized, input_var_name):
    """
    Plot the bias (bottom panel) and uncertainty (top panel )grouped by a certain input feature.
    The input feature must take discreet value for the groupedby function to work properly
    :param pred_trainval: Quantile predictions dataframe (N,M). N being the number of samples, M being the number
    of different forecast quantiles
    :param input_var_discretized: The input features used for grouping. Must be discrete values.
    :param input_var_name: name of the input variable for use in labeling
    :return fig, axarr: matplotlib Fig and axes array that contained the finished plots
    """

    # group the uncertainty and bias based on discretized input variable
    pred_trainval_groupedby_input = pred_trainval.groupby(input_var_discretized.values).mean()

    # Prepare fig. Upper for uncertainty and lower for bias.
    fig, axarr = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    # create color gradients based on E3 colors
    num_PI_pairs = (len(pred_trainval_groupedby_input.columns) - 1) / 2
    E3_colors_gradient = get_color_gradient(E3_colors, num_PI_pairs)

    # Bottom panel: Bias. Cycle through all target percentile
    for i, PI in enumerate(pred_trainval_groupedby_input.columns):
        if PI == 0.5:  # plot median forecast as a line
            axarr[1].plot(pred_trainval_groupedby_input.index, pred_trainval_groupedby_input[0.5],
                          label="Quantile = {:.1%}".format(0.5), color=E3_colors[1])
        elif PI < 0.5:  # plot symmetrical non median quantiles as a shaded range
            axarr[1].fill_between(pred_trainval_groupedby_input.index, pred_trainval_groupedby_input[PI],
                                  pred_trainval_groupedby_input[1 - PI], color=E3_colors_gradient[0, i],
                                  label="Quantile: {:.1%} to  {:.1%}".format(PI, 1 - PI))

    # axarr[1].scatter(input_trainval["Solar_RTPD_Forecast_T+1"], output_trainval, color = E3_colors[2], alpha = 0.005)
    # set labels and legends
    axarr[1].set_ylabel('Quantile of Forecast Err (MW)')
    axarr[1].legend(frameon=False, loc='center left', bbox_to_anchor=[1, 0.5])

    # top panel: uncertainty. Cycle throught all target quantiles
    for i, PI in enumerate(pred_trainval_groupedby_input.columns):
        if PI < 0.5:
            label = "P{:.0f}".format(100 * (1 - 2 * PI))
            # upward reserve is floored at 0, while downward reserve is capped at 0
            upward_reserve = np.maximum(pred_trainval_groupedby_input[1 - PI], 0)
            downward_reserve = np.minimum(pred_trainval_groupedby_input[PI], 0)

            # obtain the expected width of each bar of the bar plot
            input_var_range = pred_trainval_groupedby_input.index
            bar_width = ((input_var_range[-1] - input_var_range[0])
                         / (input_var_range.size - 1) * 0.8)

            # plot a bar for each bin of the input variable
            axarr[0].bar(input_var_range, upward_reserve, color=E3_colors_gradient[0, i],
                         width=bar_width, label=label + " Up Reserve")
            axarr[0].bar(input_var_range, downward_reserve, color=E3_colors_gradient[1, i],
                         width=bar_width, label=label + " Down Reserve")

    # labels, legends, tiles
    axarr[0].legend(frameon=False, loc='center left', bbox_to_anchor=[1, 0.5])
    axarr[0].set_ylabel('Reserves (MW)')
    axarr[0].set_title('Forecast uncertainty v.s. ' + input_var_name)
    axarr[-1].set_xlabel(input_var_name)

    fig.set_size_inches(8, 6)  # resetting figure size
    fig.tight_layout()  # avoid overlapping axes

    return fig, axarr


def discretize_input(input_var, n_bins=50):
    """
    Discretize an array that's taking continuous value into discrete values. This is done by binning the input array,
    and taking the mean of each bin as to replace all values in that bin.
    :param input_var: a numpy array or pd series.
    :param n_bins: the number of bins (dscretized values) we would like the discretized array to have
    :return input_var_discretized: discretized input var array.
    """

    # set up bins based on min, max and number of bins
    input_var_bins = np.linspace(input_var.min(), input_var.max(), n_bins)
    # replace all values in each bin with the median of that bin
    input_var_labels = (input_var_bins[:-1] + input_var_bins[1:]) / 2
    input_var_discretized = pd.cut(input_var, bins=input_var_bins, precision=0,
                                   labels=input_var_labels, include_lowest=True)

    return input_var_discretized


def get_end_metrics(training_hist, val_loss_idx=2):
    """
    Assemble the the model's metrics at the end of each training session with the training history.
    :param training_hist:  an np array of (num_PI,num_CV_folds,num_max_epoch, num_metric). For each element, it's the
    value of a certain metric for a particular PI and fold in a certain epoch. If the training finished before
    max_epoch, the rest of the array is filled with nan.
    :param val_loss_idx: we use validation loss to identify when the model training ends. This argument denote
    the position of the validation loss metric in the last dimension of training_hist.
    :return end_metrics: An np array of (num_PI, num_CV_folds, num_metric), only gather the ending (best model's)
    metrics for each PI and CV folds.
    """
    best_epoch_idx = np.nanargmin(training_hist[:, :, :, val_loss_idx], axis=2)
    end_metrics = np.take_along_axis(training_hist, best_epoch_idx[:, :, None, None], axis=2).squeeze()

    return end_metrics


def plot_compare_train_val(training_hist, PI_percentiles, metrics_to_idx_map, metrics_to_compare, x_jitter=0.1):
    """
    Visualization of the difference in loss and coverage probability between training and validation set, and also
    between different cross validation folds.
    :param training_hist: an np array of (num_PI,num_CV_folds,num_max_epoch, num_metric). Each element is the
    value of a metric for a particular PI and fold in one epoch. If the training finished before max_epoch,
    the rest of the third dimension is filled with nan.
    :param PI_percentiles: np array of (num_PI), all the target quantiles
    :param metrics_to_idx_map: A mapping between metrics' name, e.g. 'Loss', to its index in the 4th dim of the training
     hist array. Note that only the training ones' mapping are needed, since validation ones can be inferred.
    :param metrics_to_compare: List of strings. A list of metrics that we want to incorporate in the graph. Must be a
    subset of the keys of metrics_to_idx_map.
    :param x_jitter: Float. A small amount of jitter in the horizontal direction to separate training from validation
    :return: fig, axarr: Finished comparison plot.
    """

    num_PIs, num_cv_folds, _, num_metrics = training_hist.shape
    # By default, TF first record train metrics, then validation. So for the same metrics, validation is away 
    # from the training metrics by half the total amount of metrics
    train_val_metrics_dist = int(num_metrics / 2)

    # confirm that all the metrics to plot are given position
    assert set(metrics_to_compare).issubset(set(metrics_to_idx_map.keys())), "Not all metrics' position are given!"

    # Get the ending metrics, the training end is defined by the minimum in validation loss
    end_metrics = get_end_metrics(training_hist, metrics_to_idx_map['Loss (MW)'] + train_val_metrics_dist)

    # Initialize sub panels based on the number of metrics to compare
    fig, axarr = plt.subplots(1, len(metrics_to_compare), sharex=True)
    # Jitter the x position slightly to separate training and validation performance
    x_pos = np.expand_dims(np.arange(num_PIs), axis=-1) * np.ones((1, num_cv_folds))
    train_x_pos, val_x_pos = x_pos - x_jitter, x_pos + x_jitter

    # Cycle through all metrics to plot, and both the training and validation dataset
    for i, metrics in enumerate(metrics_to_compare):
        # In the default storing order of tensorflow, a metrics on training is one index 
        # before the same metrics on validation
        train_metrics_idx = metrics_to_idx_map[metrics]
        val_metrics_idx = train_metrics_idx + train_val_metrics_dist

        for j, dataset in enumerate(['Training', 'Validation']):
            x_pos = train_x_pos if dataset == 'Training' else val_x_pos
            metrics_idx = train_metrics_idx if dataset == 'Training' else val_metrics_idx

            axarr[i].scatter(x_pos.ravel(), end_metrics[:, :, metrics_idx].ravel(),
                             label=dataset, color=E3_colors[j], alpha=0.5)

        # Setting the xticks location, and the label is just the target percentiles
        axarr[i].set_xticks(np.arange(num_PIs))
        axarr[i].set_xticklabels(PI_percentiles)

        # Adding x and y axis lael
        axarr[i].set_xlabel('Tau (Target Percentile) (%)')
        axarr[i].set_ylabel(metrics)

        # For the CP metrics, add auxiliary horizontal line to denote target CP
        if metrics == 'Coverage Probability (%)':
            for PI in PI_percentiles:
                axarr[i].axhline(PI, dashes=[2, 2], color='k')

    # Legend and overall formatting
    axarr[-1].legend(loc='center left', bbox_to_anchor=[1, 0.5], frameon=False)
    fig.set_size_inches(10, 4)
    fig.tight_layout()

    return fig, axarr


def plot_example_ts(ts_ranges, pred_trainval, output_trainval):
    """
    Visualize the example periods of time, with the true errors and different quantile forecasts.
    :param ts_ranges: A list of periods to plot. Each element must be valid index for a pd.datetimeindex.
    :param pred_trainval: pd dataframe of (num_samples, num_PIs). A record of quantile forecast for all training samples
    :param output_trainval: pd dataframe of (num_samples, 1). True forecast errors for all training samples
    :return fig, axarr: Finished plots of the example timeseries. Different periods are in its own panel.
    """
    fig, axarr = plt.subplots(len(ts_ranges), 1, sharey=True)

    # create color gradients based on E3 colors
    PI_percentiles = pred_trainval.columns
    num_PI_pairs = (len(PI_percentiles) - 1) / 2
    E3_colors_gradient = get_color_gradient(E3_colors, num_PI_pairs)

    # cycle through the example time series that needs to be plotted
    for i, ts_range in enumerate(ts_ranges):

        # plot true response and median forecast
        axarr[i].plot(output_trainval.loc[ts_range], color=E3_colors[1], label='True forecast error')
        axarr[i].plot(pred_trainval.loc[ts_range, 0.5], color=E3_colors[0], dashes=[2, 2], label='Median bias - 50%')

        # cycle through each target quantile pair
        for j, PI in enumerate(PI_percentiles):
            if PI < 0.5:
                axarr[i].fill_between(pred_trainval.loc[ts_range].index, pred_trainval.loc[ts_range, PI],
                                      pred_trainval.loc[ts_range, 1 - PI], color=E3_colors_gradient[0, j],
                                      label='{:.0%}-{:.0%} range'.format(PI, 1 - PI))

        # Mark the date of this series.
        axarr[i].text(0.05, 0.85, pd.Timestamp(ts_range).strftime('%x'), transform=axarr[i].transAxes)
        axarr[i].set_ylabel("Net load forecast error (MW)")
        axarr[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axarr[i].legend(loc='center left', bbox_to_anchor=[1, 0.5], frameon=False)

    # sizing and general formatting.
    fig.set_size_inches(9, 3 * len(ts_ranges))
    fig.tight_layout()

    return fig, axarr

# ==== Unused/Archived diagnostics/plots ====

# Line plots of training history and CP change of a specific tau/fold
# def plot_training_history(history):
#     fig, axarr = plt.subplots(1,2, sharex = True)
#     history_eg = history[(0.25,0)].history

#     axarr[0].plot(history_eg['loss'],label = 'Training')
#     axarr[0].plot(history_eg['val_loss'], label = 'Validation')

#     axarr[0].set_xlabel('Epochs (#)')
#     axarr[0].set_ylabel('Losses (MW)')

#     axarr[1].plot(history_eg['CP'], label = 'Training')
#     axarr[1].plot(history_eg['val_CP'], label = 'Validation')
#     axarr[1].axhline(0.25, label = 'Target CP', dashes = [2,2], color = 'k')
#     axarr[1].set_xlabel('Epochs (#)')
#     axarr[1].set_ylabel('Coverage Probability (%)')

#     axarr[1].legend(loc = 'center left', bbox_to_anchor = [1,0.5], frameon = False)
#     fig.set_size_inches(10,4)
#     fig.tight_layout()
#     fig.savefig('training_history.png', bbox_inches = 'tight')
#     return fig, axarr


# Produce violin plots of forecast error for each bin of a certain variable
# def plot_binned_by_input_violin(pred_trainval):
#     fig, ax = plt.subplots()

#     for i,PI in enumerate([0.25,0.75]):
#         pred_trainval_gb = pred_trainval.groupby(solar_gen_bin.values)
#         pred_trainval_groupedby_solar = [pred_trainval_gb.get_group(cat)[PI] for cat in pred_trainval_gb.groups]
#         ax.boxplot(pred_trainval_groupedby_solar, showfliers= False, widths = 0.2, manage_ticks = False, patch_artist= True,
#                    boxprops={"color":"C"+str(i)}, positions= np.arange(len(bin_edges)-1)+(PI-0.5)/2) 


#     ax.set_ylabel('Quantiles (MW)')
#     ax.set_xlabel('solar geneneration (MW)')
#     ax.legend()

#     fig.set_size_inches(10,4)
#     return fig, ax


# Produce Boxplots of forecast error for each bin of a certain variable
# def plot_binned_by_input_boxplots(pred_trainval):
#     fig, ax = plt.subplots()

#     for i,PI in enumerate([0.25,0.75]):
#         for solar_bin in pred_trainval['solar_gen_bin'].unique():
#             violin_patches = ax.violinplot(pred_trainval.loc[pred_trainval['solar_gen_bin'] == solar_bin, PI],
#                                          positions = [solar_bin+(PI-0.5)/5], showmedians = False, showextrema = False)

#             for p in violin_patches['bodies']:
#                 p.set_facecolor(colors[i])
#                 p.set_alpha(0.3)

#     ax.set_ylabel('Quantiles (MW)')
#     ax.set_xlabel('solar geneneration (MW)')
#     ax.legend()

#     fig.set_size_inches(10,4)
#     return fig, ax
