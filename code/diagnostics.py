import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#==== Constants ====
E3_colors = [[3,78,110],[175,126,0],[175,34,0],[0,126,51],[175,93,0],[10,25,120]] 
E3_colors = np.array(E3_colors)/255 # E3 color pallete

#==== Active diagnostics functions in use ====
def get_color_gradient(colors, num_gradient):
    brightness_gradient = np.expand_dims((np.arange(num_gradient)+1)/(num_gradient+1), axis=-1)
    color_gradient = 1- np.expand_dims((1-colors), axis=1)*brightness_gradient
    
    return color_gradient

def plot_uncertainty_groupedby_feature(pred_trainval, input_var_discretized, input_var_name):
    
    # group the uncertainty and bias based on discretized input variable
    pred_trainval_groupedby_input = pred_trainval.groupby(input_var_discretized.values).mean()
    
    # Prepare fig. Upper for uncertainty and lower for bias.
    fig, axarr  = plt.subplots(2,1, sharex = True, gridspec_kw={"height_ratios":[1,1]})  
    # create color gradients based on E3 colors
    num_PI_pairs = (len(pred_trainval_groupedby_input.columns)-1)/2
    E3_colors_gradient = get_color_gradient(E3_colors, num_PI_pairs)
    
    # cycle through all target percentile
    for i,PI in enumerate(pred_trainval_groupedby_input.columns):
        if PI == 0.5:
            axarr[1].plot(pred_trainval_groupedby_input.index, pred_trainval_groupedby_input[0.5], 
                          label =  "Quantile = {:.1%}".format(0.5), color = E3_colors[1]) 
        elif PI<0.5:
            axarr[1].fill_between(pred_trainval_groupedby_input.index, pred_trainval_groupedby_input[PI], 
                                  pred_trainval_groupedby_input[1-PI], color =  E3_colors_gradient[0,i],
                                  label =  "Quantile: {:.1%} to  {:.1%}".format(PI, 1-PI))

    
    #axarr[1].scatter(input_trainval["Solar_RTPD_Forecast_T+1"], output_trainval, color = E3_colors[2], alpha = 0.005)
    
    axarr[1].set_ylabel('Quantile of Forecast Err (MW)')
    axarr[1].legend(frameon = False, loc = 'center left', bbox_to_anchor = [1,0.5])

    for i,PI in enumerate(pred_trainval_groupedby_input.columns):
        if PI<0.5:
            label = "P{:.0f}".format(100*(1-2*PI))
            upward_reserve = np.maximum(pred_trainval_groupedby_input[1-PI],0)
            downward_reserve = np.minimum(pred_trainval_groupedby_input[PI],0)
            
            # obtain the expected width of each bar
            input_var_range = pred_trainval_groupedby_input.index
            bar_width = ((input_var_range[-1] - input_var_range[0])
                         /(input_var_range.size - 1)*0.8)
            
            axarr[0].bar(input_var_range, upward_reserve, color = E3_colors_gradient[0,i],
                         width = bar_width, label = label+" Up Reserve")
            axarr[0].bar(input_var_range, downward_reserve,  color = E3_colors_gradient[1,i],
                         width = bar_width, label = label+" Down Reserve")

    axarr[0].legend(frameon = False, loc = 'center left', bbox_to_anchor = [1,0.5])
    axarr[0].set_ylabel('Reserves (MW)')

    axarr[0].set_title('Forecast uncertainty v.s. '+ input_var_name)
    axarr[-1].set_xlabel(input_var_name)

    fig.set_size_inches(8,6)
    fig.tight_layout()
    
    return fig, axarr

def discretize_input(input_var, n_bins = 50):
    input_var_bins = np.linspace(input_var.min(), input_var.max(), n_bins)
    input_var_labels = (input_var_bins[:-1] + input_var_bins[1:])/2 
    input_var_discretized =  pd.cut(input_var, bins = input_var_bins, precision = 0,
                                    labels = input_var_labels, include_lowest= True)
    
    return input_var_discretized

# reassemble the the model's performance after training
def get_end_metrics(training_hist, val_loss_idx=2):
    best_epoch_idx = np.nanargmin(training_hist[:,:,:,val_loss_idx], axis=2)
    end_metrics = np.take_along_axis(training_hist, best_epoch_idx[:,:, None, None], axis=2).squeeze()
    
    return end_metrics


# Visualization of the training process
def plot_compare_train_val(training_hist, PI_percentiles, 
                           metrics_to_idx_map, metrics_to_compare, x_jitter =0.1):

    num_PIs, num_cv_folds, _, num_metrics = training_hist.shape
    # By default, TF first record train metrics, then validation. So for the same metrics, validation is away 
    # from the training metrics by half the total amount of metrics
    train_val_metrics_dist = int(num_metrics/2) 

    # Get the ending metrics, the training end is defined by the minimum in validation loss
    end_metrics = get_end_metrics(training_hist, metrics_to_idx_map['Loss (MW)']+train_val_metrics_dist)
    
    # Initialize subpanel based on the number of metrics to comare
    fig, axarr = plt.subplots(1,len(metrics_to_compare), sharex = True)
    # Jitter the x position slightly to separate training and validation performance
    x_pos = np.expand_dims(np.arange(num_PIs), axis = -1)*np.ones((1,num_cv_folds)) 
    train_x_pos, val_x_pos = x_pos-x_jitter, x_pos+x_jitter 

    # Cycle through all metrics to plot, and both the training and validation dataset
    for i,metrics in enumerate(metrics_to_compare):
        # In the default storing order of tensorflow, a metrics on training is one index 
        # before the same metrics on validation
        train_metrics_idx = metrics_to_idx_map[metrics]
        val_metrics_idx = train_metrics_idx + train_val_metrics_dist

        for j,dataset in enumerate(['Training', 'Validation']):
            x_pos = train_x_pos if dataset == 'Training' else val_x_pos
            metrics_idx = train_metrics_idx if dataset == 'Training' else val_metrics_idx

            axarr[i].scatter(x_pos.ravel(), end_metrics[:,:, metrics_idx].ravel(), 
                             label = dataset, color = E3_colors[j], alpha = 0.5)

        # Setting the xticks location, and the label is just the target percentiles
        axarr[i].set_xticks(np.arange(num_PIs))
        axarr[i].set_xticklabels(PI_percentiles)

        # Adding x and y axis lael
        axarr[i].set_xlabel('Tau (Target Percentile) (%)')
        axarr[i].set_ylabel(metrics)

        # For the CP metrics, add auxilliary horizontal line to denote target CP
        if metrics == 'Coverage Probability (%)':
            for PI in PI_percentiles:
                axarr[i].axhline(PI, dashes = [2,2], color = 'k')

    # Legend and overall formatting
    axarr[-1].legend(loc = 'center left', bbox_to_anchor = [1,0.5], frameon = False)
    fig.set_size_inches(10,4)
    fig.tight_layout()
    
    return fig, axarr

# 
def plot_example_ts(ts_ranges, pred_trainval, output_trainval):
    fig, axarr = plt.subplots(len(ts_ranges),1, sharey= True)

    # create color gradients based on E3 colors
    PI_percentiles = pred_trainval.columns
    num_PI_pairs = (len(PI_percentiles)-1)/2
    E3_colors_gradient = get_color_gradient(E3_colors, num_PI_pairs)

    for i, ts_range in enumerate(ts_ranges):

        # plot true response and median forecast
        axarr[i].plot(output_trainval.loc[ts_range], color = E3_colors[1], label = 'True forecast error')
        axarr[i].plot(pred_trainval.loc[ts_range, 0.5], color = E3_colors[0], dashes = [2,2], label = 'Median bias - 50%')


        for j,PI in enumerate(PI_percentiles):
            if PI<0.5:
                axarr[i].fill_between(pred_trainval.loc[ts_range].index, pred_trainval.loc[ts_range, PI], pred_trainval.loc[ts_range,1-PI],
                                color = E3_colors_gradient[0,j], label = '{:.0%}-{:.0%} range'.format(PI, 1-PI))


        axarr[i].text(0.05,0.85, pd.Timestamp(ts_range).strftime('%x'), transform = axarr[i].transAxes)
        axarr[i].set_ylabel("Net load forecast error (MW)")
        axarr[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axarr[i].legend(loc = 'center left', bbox_to_anchor = [1,0.5], frameon = False)

    fig.set_size_inches(9,3*len(ts_ranges))
    fig.tight_layout()
    
    return fig, axarr



#==== Unused/Archived diagnostics/plots ====

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
