import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==== Constants ====
E3_colors = [[3, 78, 110], [175, 126, 0], [175, 34, 0], [0, 126, 51], [175, 93, 0], [10, 25, 120]]
E3_colors = np.array(E3_colors) / 255  # E3 color palette
HRS_IN_DAY = 24
MEDIAN = 0.5

# ==== Active diagnostics functions in use ====
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

def plot_ground_truth(trainval_outputs_by_quantile_df, true_reserves_df, q_quant_reg_rtpd_reserves_df,
                      fig, ax, quantiles_of_true_data):
    """
    Plots ground truth, both the response model was trained to predict and actual historical reserves.
    :param trainval_outputs_by_quantile_df: DataFrame of size (N, 3) - N could correspond to any time span. The 3
    columns correspond to the 3 most important quantiles, low, mid and high in order. Mid will likely be the median
    while low and high determine amt of reserves to be held given expected coverage
    :param true_reserves_df: Either None or a DataFrame of size (N, 2) - N must correspond to the same time span as the
    trainval_outputs. The 2 columns hold upward and downward reserves held in reality
    :param fig, ax: The figure and axis objects to be plotted on top of, in this function
    :param quantiles_of_true_data: List(Float) of size (3) - The low, mid and high quantiles as previously described
    :return: fig, ax: Figure and axis objects that contain the ground-truth plots made in this function
    """
    lower_quantile_of_true_data, mid_quantile_of_true_data, upper_quantile_of_true_data = quantiles_of_true_data
    # Plot trainval outputs for each quantile
    ax.scatter(trainval_outputs_by_quantile_df.index, trainval_outputs_by_quantile_df[mid_quantile_of_true_data],
               label="Ground Truth, Quantile = {:.1%}".format(mid_quantile_of_true_data),
               color=E3_colors[2], alpha=0.5)
    ax.scatter(trainval_outputs_by_quantile_df.index, trainval_outputs_by_quantile_df[lower_quantile_of_true_data],
               label="Ground Truth, Quantile: {:.1%} to {:.1%}".format(lower_quantile_of_true_data,
                                                                       upper_quantile_of_true_data),
               color=E3_colors[5], alpha=0.5)
    ax.scatter(trainval_outputs_by_quantile_df.index, trainval_outputs_by_quantile_df[upper_quantile_of_true_data],
               color=E3_colors[5], alpha=0.5)
    # Plot true reserves data, if provided by user
    if true_reserves_df is not None:
        ax.plot(true_reserves_df.index, true_reserves_df["UP"], label="CAISO's Hist Method, 2.5% to 97.5%", color=E3_colors[3])
        ax.plot(true_reserves_df.index, true_reserves_df["DOWN"], color=E3_colors[3])
    if q_quant_reg_rtpd_reserves_df is not None:
        ax.plot(q_quant_reg_rtpd_reserves_df.index, q_quant_reg_rtpd_reserves_df["UP"], label="CAISO's Q-Regression, 2.5% to 97.5%",
                color=E3_colors[3], linestyle = "--")
        ax.plot(q_quant_reg_rtpd_reserves_df.index, q_quant_reg_rtpd_reserves_df["DOWN"], color=E3_colors[3],
                linestyle = "--")
    return fig, ax

def plot_model_predictions(model_pred_df, fig, ax):
    """
    Plots model predictions at different quantiles
    :param model_pred_df: Quantile predictions dataframe (N,M). N being the number of samples, M being the number
    of different forecast quantiles
    :param fig, ax: The figure and axis objects to be plotted on top of, in this function
    :return: fig, ax: Figure and axis objects that contain the model predictions plotted in this function
    """
    # Find the total number of prediction interval pairs
    num_PI_pairs = (len(model_pred_df.columns) - 1) / 2
    
    # Get colors for different prediction intervals to be plotted
    E3_colors_gradient = get_color_gradient(E3_colors, num_PI_pairs)

    # Plot each prediction interval
    for i, PI in enumerate(model_pred_df.columns):
        if PI == MEDIAN:  # plot median forecast as a line
            ax.plot(model_pred_df.index, model_pred_df[MEDIAN],
                          label="E3 Prediction,\nQuantile: {:.1%}".format(MEDIAN), color=E3_colors[1])
        elif PI < MEDIAN:  # plot symmetrical non median quantiles as a shaded range
            ax.fill_between(model_pred_df.index, model_pred_df[PI],
                                  model_pred_df[1 - PI], color=E3_colors_gradient[0, i],
                                  label="E3 Prediction,\nQuantile: {:.1%} to  {:.1%}".format(PI, 1 - PI))

    return fig, ax

# def groupby_helper(list_of_df, input_var_discretized, list_of_agg_func, list_of_interp):
#     """
#     Performs groupby and aggregate operations on a list of DataFrames
#     :param list_of_df: List of DataFrames operations are to be performed on
#     :param input_var_discretized: Discretized variable groups used to perform groupby
#     :param list_of_agg_func: How should groups be aggregated after groupby? Options are eihter "mean" or a float
#     corresponding to the quantile value that will represent each group
#     :param list_of_interp: If quantile isn't unique, what "interpolation" method is to be used to pick one?
#     :return: List of DataFrames after groupby and aggregation has been performed on each
#     """
#     # Initialize collector to store dfs after groupby and aggregation by user-defined func is conducted
#     list_of_grouped_df = []
#     # Iterate over each df, aggregating as user desires
#     for df_idx, df in enumerate(list_of_df):
#         if list_of_agg_func[df_idx] == "mean":
#             list_of_grouped_df.append(df.groupby(input_var_discretized).mean())
#         else:
#             list_of_grouped_df.append(df.groupby(input_var_discretized).quantile(list_of_agg_func[df_idx],
#                                                                              interpolation = list_of_interp[df_idx]))
#     return list_of_grouped_df

def find_datetimes(df, master_df):
    """
    Identifies a datetime corresponding to each entry in df by looking for that entry in the master df
    :param df: DataFrame with entries, corresponding datetimes for which need to be identified
    :param master_df: DataFrame with datetimes as index and a single column holding entries which must comprise
    df's entries within it
    :return: A DataFrame consistent in shape as df, holding datetimes corresponding to df's entries in the master df
    """
    # Define collector to store these datetimes
    datetimes_df = pd.DataFrame(index = df.index, columns = df.columns, dtype = object)
    # Iterate over each column and row(hour), finding the datetime corresponding to each element in df
    # and storing it in datetimes df
    for col in df.columns:
        for hr in df.index:
            master_df_in_hour = master_df[master_df.index.hour == hr]
            bool_arr = master_df[master_df.index.hour == hr].values == df.loc[hr, col]
            date_time = master_df[master_df.index.hour == hr].index[bool_arr][0] # Pick first if multiple
            datetimes_df.loc[hr, col] = date_time

    return datetimes_df

"""
    :param hist_rtpd_reserves: pd.DataFrame (N,2), N being the number of samples and 2-> Up and Down reserves. Contains
    actual reserves data, if provided by the user to be visually compared to true forecast errors and model predictions
    

    # If the response is associated with net load, historical FRP reserves are relevant. Identify those at
    # datetimes identified above for each quantile if user passed on the reserves info
    if hist_rtpd_reserves is None:
        q_hist_rtpd_reserves = None
    else:
        # Forecast error is defined as Forecast - Actual. So, need UP reserves when forecast error is negative
        # i.e Model predictions for quantiles below MEDIAN correspond to UP reserves needed
        q_hist_rtpd_reserves_up = hist_rtpd_reserves.loc[q_datetimes_df[lower_quantile_of_true_data].values, "UP"].values
        q_hist_rtpd_reserves_down = hist_rtpd_reserves.loc[q_datetimes_df[upper_quantile_of_true_data].values, "DOWN"].values
        # Multiply DOWN reserves by (-1) so they will be plotted below reserves = 0 line in the figure
        # This script assumes both UP and DOWN reserves are positive in the raw data being fed
        q_hist_rtpd_reserves_down *= (-1)
        q_hist_rtpd_reserves = pd.DataFrame({"UP":q_hist_rtpd_reserves_up, "DOWN":q_hist_rtpd_reserves_down})

    if quant_reg_rtpd_reserves is None:
        q_quant_reg_rtpd_reserves = None
    else:
        # Forecast error is defined as Forecast - Actual. So, need UP reserves when forecast error is negative
        # i.e Model predictions for quantiles below MEDIAN correspond to UP reserves needed
        q_quant_reg_rtpd_reserves_up = quant_reg_rtpd_reserves.loc[q_datetimes_df[lower_quantile_of_true_data].values, "UP"].values
        q_quant_reg_rtpd_reserves_down = quant_reg_rtpd_reserves.loc[q_datetimes_df[upper_quantile_of_true_data].values, "DOWN"].values
        # Multiply DOWN reserves by (-1) so they will be plotted below reserves = 0 line in the figure
        # This script assumes both UP and DOWN reserves are positive in the raw data being fed
        q_quant_reg_rtpd_reserves_down *= (-1)
        q_quant_reg_rtpd_reserves = pd.DataFrame({"UP":q_quant_reg_rtpd_reserves_up,
                                                  "DOWN":q_quant_reg_rtpd_reserves_down})
"""


def compare_predictions_to_truth(pred_trainval, input_var_name, response_label, response_type,
                                 trainval_outputs):
    """
    Plot the the model predictions and the ground truth for points that lie at the very edge of the desired prediction 
    interval in each hour. True reserves held can be overlaid on top for comparison with model predicted reserves if
    provided by user.
    :param pred_trainval: Quantile predictions dataframe (N,M). N being the number of samples, M being the number
    of different forecast quantiles
    
    :param input_var_name: str, name of the input variable for use in labeling
    :param response_label: str,name of the output variable for use in labeling
    :param response_type: str, = "load" or "generation" based on model-response type. Used to determine sign-convention
    to ensure headroom and footroom are always above reserves = 0 and below reserves = 0 axis respectively
    :param trainval_outputs: pd.Series (N,1), N being the number of samples. Contains the ground truth the model was
    trained to predict

    :return fig, axarr: matplotlib Fig and axes array that contains the finished plots
    :return q_truth_df, q_model_pred_df: DataFrame(FLoats) - Hold truth/model prediction data for each hour, as being
    plotted in this function. Passed out for user to analyze and save as need be.
    """
    
    # unpack the list of quantiles where coincident comparison happens
    quantiles_list = pred_trainval.columns 
    lower_quantile, upper_quantile = quantiles_list 

    # Identify datetimes wherein true forecast error was at user defined quantile(s) in each hour
    # Since we need an exact datetime, can't interpolate between quantiles.
    # "Interpolate" such that the quantile doesn't go out of desired prediction interval coverage.
    # When trying to find 2.5th quantile, we'll settle for 3rd quantile and when finding
    # 97.5th quantile, we'll settle for the 97th for example
    
    list_of_interp_method = ["lower", "higher"]
            
    # Find true forecast error at a particular quantile in each hour
    q_truth_df = pd.concat(groupby_helper([trainval_outputs, trainval_outputs, trainval_outputs],
                                          trainval_outputs.index.hour, pred_trainval.columns, list_of_interp_method),
                           axis = 1)
    
    q_truth_df.columns = quantiles_list

    # Find datetimes corresponding to these forecast errors
    q_datetimes_df = find_datetimes(q_truth_df, trainval_outputs)
    
    # Get the model prediction at the datetimes identified above
    q_model_pred_df = pd.DataFrame(index = np.arange(HRS_IN_DAY), columns = quantiles_list)
    
    for quant in quantiles_of_true_data:
        # Retrieve datetimes for this quant
        dates_at_quant = q_datetimes_df[quant].values
        # Retrieve predictions at these datetimes
        model_pred_at_quant = pred_trainval.loc[dates_at_quant, quant].values
        # Store predictions - to be plotted later
        q_model_pred_df[quant] = model_pred_at_quant
    
    #TODO: Flip signs based on response types
    
    # Plot the true forecast errors, coincident model predictions and coincident FRP reserves, if applicable
    fig, ax = plt.subplots()
    
    # Plot coincident model predictions
    fig, ax = plot_model_predictions(q_model_pred_df, num_PI_pairs, fig, ax)
    
    # Plot true forecast errors and hist rtpd reserves (if provided by user)
    fig, ax = plot_ground_truth(q_truth_df, q_hist_rtpd_reserves, q_quant_reg_rtpd_reserves,
                                fig, ax, quantiles_of_true_data)
        
    # Set labels and legend
    ax.set_title(response_label + ' Forecast Uncertainty v.s. ' + input_var_name)
    ax.set_ylabel('MW of Forecast Error/Reserves')
    ax.set_xlabel('Hour of Day')
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=[1, 0.5])
    ax.set_xlim([0, HRS_IN_DAY - 1])
        
    return fig, ax, q_truth_df, q_model_pred_df

"""
Parking lot for vignesh's code

    :param hist_rtpd_reserves: pd.DataFrame (N,2), N being the number of samples and 2-> Up and Down reserves. Contains
    actual reserves data, if provided by the user to be visually compared to true forecast errors and model predictions
    :param quantiles_of_true_data: List (Float), The low, mid and high quantiles of true data to be plotted

    :param response_type: str, = "load" or "generation" based on model-response type. Used to determine sign-convention
    to ensure headroom and footroom are always above reserves = 0 and below reserves = 0 axis respectively
    :param trainval_inputs: pd.Series (N,1), N being the number of samples. Contains the input variable values that the
    response variable will be plotted against. Can be used to overlay a response-input scatter plot on top of the
    diagnostic plot
    :param trainval_outputs: pd.Series (N,1), N being the number of samples. Contains the ground truth the model was
    trained to predict
"""

"""
    # Group RTPD reserves data, if provided by user
    if hist_rtpd_reserves is None:
        hist_rtpd_reserves_groupedby_input = None
    else:
        hist_rtpd_reserves_groupedby_input = hist_rtpd_reserves.groupby(input_var_discretized.values).mean()
        # Multiply DOWN reserves by (-1) so they will be plotted below reserves = 0 line in the figure
        # This script assumes both UP and DOWN reserves are positive in the raw data being fed
        hist_rtpd_reserves_groupedby_input["DOWN"] = -hist_rtpd_reserves_groupedby_input["DOWN"]

    # Group RTPD reserves data, if provided by user
    if quant_reg_rtpd_reserves is None:
        quant_reg_rtpd_reserves_groupedby_input = None
    else:
        quant_reg_rtpd_reserves_groupedby_input = quant_reg_rtpd_reserves.groupby(input_var_discretized.values).mean()
        # Multiply DOWN reserves by (-1) so they will be plotted below reserves = 0 line in the figure
        # This script assumes both UP and DOWN reserves are positive in the raw data being fed
        quant_reg_rtpd_reserves_groupedby_input["DOWN"] = -quant_reg_rtpd_reserves_groupedby_input["DOWN"]

"""


def ensure_sign_convention():
    """
    This ensures head-room is shown above reserves = 0 axis and foot-room is conversely shown below reserves = 0 axis
    """ 
    if response_type == "load":
        pred_trainval_groupedby_input = - pred_trainval_groupedby_input
        trainval_outputs_groupedby_input = - trainval_outputs_groupedby_input
        
    return pred_trainval_groupedby_input, trainval_outputs_groupedby_input



def plot_uncertainty_groupedby_feature(pred_trainval, input_var_discretized, input_var_name, response_label):
    """
    Plot the bias (bottom panel) and uncertainty (top panel )grouped by a certain input feature.
    The input feature must take discreet value for the groupedby function to work properly
    :param pred_trainval: Quantile predictions dataframe (N,M). N being the number of samples, M being the number
    of different forecast quantiles
    :param input_var_discretized: The input feature array (N,1) used for grouping. Must be discrete values.
    :param input_var_name: str, name of the input variable for use in labeling
    :param response_label: str ,name of the output variable for use in labeling
    :return fig, axarr: matplotlib Fig and axes array that contained the finished plots

    """
    
    # Group model predictions and ground truth  based on discretized input variable
    pred_trainval_groupedby_input = pred_trainval.groupby(input_var_discretized.values).mean()

    # Prepare ax array
    fig, ax = plt.subplots()

    # Plot model predictions at all target quantiles
    fig, ax = plot_model_predictions(pred_trainval_groupedby_input, fig, ax)

    # Plot ground truth and historical rtpd reserves(if provided by user)
    #fig, ax = plot_ground_truth(trainval_outputs_groupedby_input, hist_rtpd_reserves_groupedby_input,
     #                           quant_reg_rtpd_reserves_groupedby_input, fig, ax, quantiles_of_true_data)

    # set labels and legends
    ax.set_ylabel('Quantile of Forecast Err (MW)')
    ax.legend(frameon=False, loc='center left', bbox_to_anchor=[1, 0.5])
    ax.set_title(response_label + ' Forecast Uncertainty v.s. ' + input_var_name)
    ax.set_xlabel(input_var_name)

    return fig, ax


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
    :param val_loss_idx: Validation loss is used to identify the end of model training. This argument denote
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

def plot_example_ts(ts_ranges, pred_trainval, output_trainval, response_label, response_type):
    """
    Visualize the example periods of time, with the true errors and different quantile forecasts.
    :param ts_ranges: A list of periods to plot. Each element must be valid index for a pd.datetimeindex.
    :param pred_trainval: pd dataframe of (num_samples, num_PIs). A record of quantile forecast for all training samples
    :param output_trainval: pd dataframe of (num_samples, 1). True forecast errors for all training samples
    :param response_label: str, name of the output variable for use in labeling
    :param response_type: str, = "load" or "generation" based on model-response type. Used to determine sign-convention
    to ensure headroom and footroom are always above reserves = 0 and below reserves = 0 axis respectively
    :return fig, axarr: Finished plots of the example timeseries. Different periods are in its own panel.
    """
    fig, axarr = plt.subplots(len(ts_ranges), 1, sharey=True)

    # This ensures  head-room is shown above reserves = 0 axis and foot-room is conversely shown below reserves = 0 axis
    if response_type == "load":
        pred_trainval = - pred_trainval
        output_trainval = - output_trainval

    # cycle through the example time series that needs to be plotted
    for i, ts_range in enumerate(ts_ranges):

        # plot true response and median forecast
        axarr[i].plot(output_trainval.loc[ts_range], color=E3_colors[1], label='True Forecast Error')
        axarr[i].plot(pred_trainval.loc[ts_range, MEDIAN], color=E3_colors[0], dashes=[2, 2], label='Median bias - 50%')

        # plot model predictions at each target quantile
        fig, axarr[i] = plot_model_predictions(pred_trainval.loc[ts_range], fig, axarr[i])

        # Mark the date of this series.
        axarr[i].text(0.05, 0.85, pd.Timestamp(ts_range).strftime('%x'), transform=axarr[i].transAxes)
        axarr[i].set_ylabel(response_label + " Forecast Error (MW)")
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
