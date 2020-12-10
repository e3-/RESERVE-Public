import numpy as np
import pandas as pd
import os

# Define metrics

def coverage(y_true, y_pred):
    '''

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Fraction of observed forecast errors that fall below / are "covered" by quantile estimates

    '''
    return np.mean(y_true <= y_pred)

def requirement(y_true, y_pred):
    '''

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Average reserve level/requirement, which corresponds to the average of the quantile estimates

    '''
    return np.mean(y_pred)

def exceeding(y_true, y_pred):
    '''

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Average excess of observed forecast errors above the quantile estimates when observed forecast errors exceed
            corresponding quantile estimates

    '''
    return np.mean((y_true - y_pred)[y_true > y_pred])

def closeness(y_true, y_pred):
    '''

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Average (absolute) distance between observed forecast errors and quantile estimates; equivalent to mean average
            error (MAE) between observed forecast errors and quantile estimates

    '''
    return np.mean(np.abs(y_true - y_pred))

def max_exceeding(y_true, y_pred):
    '''

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Maximum excess of observed forecast errors above corresponding quantile estimates

    '''
    return np.max(y_true - y_pred)

def reserve_ramp_rate(y_true, y_pred):
    '''

    Args:
        y_true: Time series of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model

    Returns:
        Average ramp rate of reserve level/requirement (average absolute rate of change)

    '''
    return np.mean(np.abs(y_pred.values[1:] - y_pred.values[:-1])/((y_pred.index[1:] - y_pred.index[:-1]).astype(int)/(1e9*3600)))

def pinball_risk(y_true, y_pred, tau = 0.975):
    '''

    Args:
        y_true: Time seriers of observed forecast errors
        y_pred: Time series of corresponding conditional quantile estimates from machine learning model
        tau: Target percentile for quantile estimates (needed within calculation); default = 0.975

    Returns:
        Average pinball loss of input data; similar to "closeness" metric, but samples are re-weighted so that the
            metric is minimized for "optimal" or "true" quantile estimation models

    '''
    return np.mean(np.max(np.array([(1-tau)*(y_pred - y_true), tau*(y_true - y_pred)]), axis = 0))

# Define function to compute/writeout metrics

def compute_metrics(output_trainval,
                    pred_trainval,
                    df=None,
                    tau=0.975,
                    filename=None,
                    metrics=[coverage,
                             requirement,
                             exceeding,
                             closeness,
                             max_exceeding,
                             reserve_ramp_rate,
                             pinball_risk]
                    ):

    '''

    Description:
        Iteratively computes metrics for input data and returns metrics in a pandas dataframe

    Args:
        output_trainval: Dataframe of observed forecast errors
        pred_trainval: Dataframe of corresponding conditional quantile estimates from machine learning model for
            multiple CV folds
        df: Existing dataframe containing metrics (e.g. for another tau-level); default = None
        tau: Target percentile for predictions (also an input for pinball loss metric); default = 0.975
        filename: Path to file where metrics will be saved if filename specified; default = None
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

    '''

    pinball_risk.__defaults__ = (tau,)  # Set pinball risk default tau-level to input tau (default will remain tau = 0.975 if no value is specified)
    CV_folds = np.arange(10)  # Define array of CV fold IDs

    if df is None:
        df = pd.DataFrame()  # Create new dataframe if no existing dataframe is given
        df['metrics'] = [metric.__name__ for metric in metrics]
        df.set_index('metrics', inplace=True)  # Set index to list of metrics
        df.index.name = None

    y_true = output_trainval  # Define y_true
    for j, CV in enumerate(CV_folds):
        y_pred = pred_trainval[(tau, CV)]  # Define y_pred (from tau, CV fold ID)
        df[(tau, CV)] = ""  # Create empty column to hold metrics
        for metric in metrics:
            df[(tau, CV)][metric.__name__] = metric(y_true, y_pred)  # Compute metric

    df = df.T.set_index(
        pd.MultiIndex.from_tuples(df.T.index, names=('Quantiles', 'Fold ID'))).T  # Reformat to have multi-level columns

    if filename != None:
        df.to_csv(filename)  # Writeout to CSV file

    return df


if __name__ == "__main__":

    CAISO_data = pd.read_csv(os.path.join('CAISO Metrics', 'CAISO_measurements.csv'), index_col='Unnamed: 0')

    print('Measurements reported by CAISO for Histogram method:\n')

    print('Coverage: {}%'.format(100*CAISO_data['Coverage']['Histogram']))
    print('Requirement: {} MW'.format(CAISO_data['Requirement']['Histogram']))
    print('Closeness: {} MW'.format(CAISO_data['Closeness']['Histogram']))
    print('Exceeding: {} MW'.format(CAISO_data['Exceeding']['Histogram']))

    print('\nMeasurements reported by CAISO for Quantile Regression method:\n')

    print('Coverage: {}%'.format(100*CAISO_data['Coverage']['Quantile Regression']))
    print('Requirement: {} MW'.format(CAISO_data['Requirement']['Quantile Regression']))
    print('Closeness: {} MW'.format(CAISO_data['Closeness']['Quantile Regression']))
    print('Exceeding: {} MW'.format(CAISO_data['Exceeding']['Quantile Regression']))

    print('\nRESCUE performance metrics:\n')

    output_trainval = pd.read_pickle('C:\\Users\\charles.gulian\\PycharmProjects\\RESCUE\\data\\rescue_v1_2\\output_trainval.pkl')
    pred_trainval1 = pd.read_pickle('C:\\Users\\charles.gulian\\PycharmProjects\\RESCUE\\output\\rescue_v1_1\\pred_trainval.pkl')

    df = compute_metrics(output_trainval, pred_trainval1)

    print('Coverage: {:.2f}%'.format(100 * df.loc['coverage'].mean()))
    print('Requirement: {:.2f} MW'.format(df.loc['requirement'].mean()))
    print('Closeness: {:.2f} MW'.format(df.loc['closeness'].mean()))
    print('Exceeding: {:.2f} MW'.format(df.loc['exceeding'].mean()))
    print('Max. Exceeding: {:.2f} MW'.format(df.loc['max_exceeding'].mean()))
    print('Mean Reserve Ramp Rate: {:.2f} MW/hr'.format(df.loc['reserve_ramp_rate'].mean()))
    print('Pinball Risk: {:.2f} MW'.format(df.loc['pinball_risk'].mean()))

    print('\nMetrics dataframe:\n')
    print(df)