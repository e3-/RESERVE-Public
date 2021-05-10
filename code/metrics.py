import numpy as np
import pandas as pd
import os
import pathlib


# Define helper function to get validation set predictions from cross-validation fold masks
def get_validation_preds(pred_trainval, val_masks):
    '''
    Stitch together validation set predictions from multi-objective pred_trainval predictions dataframe

    Args:
        pred_trainval (Pandas DataFrame): predictions dataframe from multi-objective RESCUE model
        tau (float): target quantile (between 0 and 1)
        output (str): dataframe column name for prediction output/target objective (e.g. net load, load, solar, wind)
        val_masks (array): numpy array containing validation set mask for each cross-validation fold in its rows

    Returns:
        preds (Pandas Series): Validation set predictions for given tau, target from all CV folds stitched together along a single axis
    '''

    # check if the validation masks covers exactly once the full range of trainval data
    assert (np.sum(val_masks, axis=0) == 1).all(), "the validation masks does not cover trainval data exactly once!"

    # initialize container for the prediction on the validation set alone
    pred_val = pred_trainval.groupby(axis=1, level=['Quantiles', 'Output_Name']).mean()
    pred_val[:] = 0

    # cycle through all tau and outputs
    for tau in pred_trainval.columns.get_level_values('Quantiles').unique():
        for output in pred_trainval.columns.get_level_values('Output_Name').unique():
            curr_pred = pred_trainval.xs((tau, output), axis=1, level=['Quantiles', 'Output_Name'])
            pred_val[(tau, output)] = (curr_pred * val_masks.T).sum(axis=1)

    return pred_val


# Define metrics

def coverage(y_true, y_pred, **kwargs):
    """

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Fraction of observed forecast errors that fall below / are "covered" by quantile estimates

    """
    return (y_true <= y_pred).mean()


def requirement(y_true, y_pred, **kwargs):
    """

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Average reserve level/requirement, which corresponds to the average of the quantile estimates

    """
    return y_pred.mean()


def closeness(y_true, y_pred, **kwargs):
    """

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Average (absolute) distance between observed forecast errors and quantile estimates; equivalent to mean average
            error (MAE) between observed forecast errors and quantile estimates

    """
    return (y_true - y_pred).abs().mean()


def exceedance(y_true, y_pred, tau=0.975, **kwargs):
    """

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Average excess of observed forecast errors above (or below) the quantile estimates when observed forecast errors
        exceed corresponding quantile estimates

    """
    if tau >= 0.5:
        return ((y_true - y_pred)[y_true >= y_pred]).mean()
    else:
        return ((y_true - y_pred)[y_true <= y_pred]).mean()


def max_exceedance(y_true, y_pred, tau=0.975, **kwargs):
    """

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Maximum excess of observed forecast errors above (or below) corresponding quantile estimates

    """
    if tau >= 0.5:
        return (y_true - y_pred).max()
    else:
        return (y_true - y_pred).min()


def pinball_loss(y_true, y_pred, tau=0.975, **kwargs):
    '''

    Args:
        y_true: Time seriers of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model
        tau: Target percentile for quantile estimates (needed within calculation); default = 0.975

    Returns:
        Average pinball loss of input data; similar to "closeness" metric, but samples are re-weighted so that the
            metric is minimized for "optimal" or "true" quantile estimation models

    '''
    #     mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    #     y_true, y_pred = y_true[mask], y_pred[mask]
    return np.mean(np.max(np.array([(1 - tau) * (y_pred - y_true), tau * (y_true - y_pred)]), axis=0))


def reserve_ramp_rate(y_true, y_pred, **kwargs):
    '''

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Average ramp rate of reserve level/requirement (average absolute rate of change)

    '''
    return np.mean(np.abs(
        (y_pred.values[1:] - y_pred.values[:-1]) / ((y_pred.index[1:] - y_pred.index[:-1]).total_seconds() / 3600)))


# Define function to compute/writeout metrics

def compute_metrics_for_specified_tau(output_trainval, pred_trainval, df=None, tau=0.975, val_masks=None,
                                      metrics=(coverage,
                                               requirement,
                                               exceedance,
                                               closeness,
                                               max_exceedance,
                                               reserve_ramp_rate,
                                               pinball_loss)):
    """

    Description:
        Iteratively computes metrics for input data and returns metrics in a pandas dataframe

    Args:
        output_trainval: Dataframe of observed forecast errors
        pred_trainval: Dataframe of corresponding conditional quantile estimates from machine learning model for
            multiple CV folds
        df: Existing dataframe containing metrics (e.g. for another tau-level); default = None
        tau: Target percentile for predictions (also an input for pinball loss metric); default = 0.975
        filename: Path to file where metrics will be saved if filename specified; default = None
        val_masks: Array containing cross-validation fold validation set masks in rows
        metrics: List of metrics to compute for input data

    Returns:
        df: Dataframe containing metrics for current value of tau (and with metrics for other values of tau if existing
            dataframe was passed to function

    Example usage:

        # Get metrics dataframe for target percentile of 97.5%
        df_metrics = compute_metrics(output_trainval, pred_trainval)

        # Get metrics dataframe for target percentiles of 95% and 97.5% by passing previously computed dataframe
        df_metrics = compute_metrics(output_trainval, pred_trainval, tau = 0.95, df = df_metrics)

        # Save metrics dataframe to "file.csv"
        compute_metrics(output_trainval, pred_trainval, filename = 'file.csv')

    """

    CV_folds = pred_trainval.columns.levels[1]  # Define array of CV fold IDs
    outputs = pred_trainval.columns.levels[2]  # Define array of target outputs (load, net load, solar, wind)

    # Initialize dataframe if not passed to function in arguments
    if df is None:
        df = pd.DataFrame(dtype='float')  # Create new dataframe if no existing dataframe is given
        df['metrics'] = [metric.__name__ for metric in metrics]
        df.set_index('metrics', inplace=True)  # Set index to list of metrics
        df.index.name = None

    # default to using entire series if validation mask is not provided
    if val_masks is None:
        val_masks = np.ones(CV_folds, len(pred_trainval))

    # cycle through each of the output
    for output in outputs:
        # cycle through each of the CV fold to calculate metric
        for j, CV in enumerate(CV_folds):

            y_true = output_trainval[output].loc[val_masks[j]]  # Define y_true (validation set)
            y_pred = pred_trainval[(tau, CV, output)].loc[val_masks[j]]  # Define y_pred (validation set)
            df[(tau, CV, output)] = ""  # Create empty column to hold metrics

            for metric in metrics:
                df[(tau, CV, output)][metric.__name__] = metric(y_true, y_pred, tau=tau)  # Compute metric

    df = df.T.set_index(
        pd.MultiIndex.from_tuples(df.T.index, names=(
        'Quantiles', 'Fold ID', 'Output_Name'))).T  # Reformat to have multi-level columns

    return df


def compute_metrics_for_all_taus(output_trainval, pred_trainval, val_masks=None,
                                 dir_str=None, avg_across_folds=True):
    """
    :param output_trainval:Dataframe of observed forecast errors
    :param pred_trainval: Dataframe of corresponding conditional quantile estimates from machine learning model for
            multiple CV folds and multiple tau. The columns are two leveled, with the sequence being (tau, CV)
    :param avg_across_folds: a boolean determining whether to return the metrics for each fold or the average
    :return: Dataframe containing metrics for all values of tau present in the pred_trainval
    """

    metrics_value_df = None

    for tau in pred_trainval.columns.levels[0]:
        metrics_value_df = compute_metrics_for_specified_tau(output_trainval, pred_trainval,
                                                             df=metrics_value_df, tau=tau, val_masks=val_masks)
    # write all metrics to hard drive 
    if dir_str is not None:
        metrics_value_df.to_pickle(dir_str.metrics_path)  # Write to pkl file

    if avg_across_folds:
        metrics_value_df = metrics_value_df.astype('float').groupby(axis=1, level=['Quantiles', 'Output_Name']).mean()

    return metrics_value_df


def n_crossings(pred_trainval, filename=None):
    """

    Computes number of quantile crossings within CV folds for various target percentile pairs

    Args:
        pred_trainval: Dataframe containing quantile estimates for each CV fold and target percentile

    Returns:
        Dataframe containing number of quantile crossings for each pair of target percentiles within each CV fold
            (only for valid pairs of "lower" and "upper" target percentiles)
    """

    tau_arr = pred_trainval.columns.levels[0]  # Define array of tau (target quantiles)
    CV_folds = pred_trainval.columns.levels[1]  # Define array of CV fold IDs
    outputs = pred_trainval.columns.levels[2]  # Define array of target outputs (load, net load, solar, wind)
    num_samples = pred_trainval.shape[0]

    crossings = {}  # Define dictionary to store crossings

    for CV in CV_folds:
        for output in outputs:
            # Look for quantile crossings only in sets of predictions from models trained on same CV fold and predicting same target output
            crossings[(CV, output)] = {}
            for i, t1 in enumerate(tau_arr):
                for j, t2 in enumerate(tau_arr[
                                       i + 1:]):  # Only evaluate number of quantile crossings on valid lower/upper target percentile pairs
                    crossings[(CV, output)][(t1, t2)] = sum(
                        pred_trainval[(t1, CV, output)] > pred_trainval[
                            (t2, CV, output)]) / num_samples  # Record number of quantile crossings

    df = pd.DataFrame(crossings)
    df.columns.names = ('CV Fold ID', 'Output_Name')
    df.index.rename(['Lower Quantile', 'Upper Quantile'], inplace=True)

    if filename != None:
        df.to_csv(filename)  # Writeout to CSV file

    return df


if __name__ == "__main__":
    import cross_val
    import utility

    model_name = 'rescue_v1_1_multi_objective'
    num_cv_folds = 10

    dir_str = utility.Dir_Structure(model_name=model_name)
    input_trainval = pd.read_pickle(dir_str.input_trainval_path)
    output_trainval = pd.read_pickle(dir_str.output_trainval_path)
    val_masks_all_folds = cross_val.get_CV_masks(input_trainval.index, num_cv_folds, dir_str.shuffled_indices_path)

    curr_dir = pathlib.Path('.')
    preds = curr_dir / '..' / 'output' / model_name / 'pred_trainval.pkl'
    pred_trainval = pd.read_pickle(preds)

    print('Metrics:')
    print(compute_metrics_for_all_taus(output_trainval, pred_trainval, val_masks=val_masks_all_folds))

    print('# Quantile Crossings:')
    print(n_crossings(pred_trainval))
