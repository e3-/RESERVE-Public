import numpy as np
import pandas as pd
import os

def coverage(y_true, y_pred):
    return np.mean(y_true <= y_pred)

def requirement(y_true, y_pred):
    return np.mean(y_pred)

def exceeding(y_true, y_pred):
    return np.mean((y_true - y_pred)[y_true > y_pred])

def closeness(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def max_exceeding(y_true, y_pred):
    return np.max(y_true - y_pred)

def reserve_ramp_rate(y_true, y_pred):
    dt = 0.25 # hours (interval between time periods)
    return np.mean([np.abs(y_pred[i] - y_pred[i-1]) for i in range(1,len(y_pred))])/dt

def n_crossings(y_pred1, y_pred2):
    # y_pred1 is lower quantile estimate; y_pred2 is upper quantile estimate (crossing when y_pred1 > y_pred2)
    return np.sum(y_pred1 > y_pred2)

if __name__ == "__main__":

    print('Be sure to activate e3rescue virtual environment!\n\n')

    CAISO_data = pd.read_csv(os.path.join('CAISO Metrics', 'CAISO_measurements.csv'), index_col='Unnamed: 0')
    data_df = pd.read_pickle(os.path.join('outputs_from_code', 'metrics_data_v2.pkl'))
    d = data_df.to_dict()

    print('Measurements reported by CAISO for Histogram Approach:\n')

    print('Coverage: {}%'.format(CAISO_data['Coverage']['Histogram']))
    print('Requirement: {} MW'.format(CAISO_data['Requirement']['Histogram']))
    print('Closeness: {} MW'.format(CAISO_data['Closeness']['Histogram']))
    print('Exceeding: {} MW'.format(CAISO_data['Exceeding']['Histogram']))

    print('\nMeasurements reported by CAISO for Quantile Regression Approach:\n')

    print('Coverage: {}%'.format(CAISO_data['Coverage']['Quantile Regression']))
    print('Requirement: {} MW'.format(CAISO_data['Requirement']['Quantile Regression']))
    print('Closeness: {} MW'.format(CAISO_data['Closeness']['Quantile Regression']))
    print('Exceeding: {} MW'.format(CAISO_data['Exceeding']['Quantile Regression']))

    q1, q2 = 0.025, 0.975
    print('\nRESCUE performance metrics:\n')

    y_true = np.concatenate([d[(q1, CV)]['y_true'] for CV in range(10)]).reshape(-1, 1)
    y_pred = np.concatenate([d[(q2, CV)]['y_pred'] for CV in range(10)])

    print('Coverage: {}%'.format(100 * coverage(y_true, y_pred)))
    print('Requirement: {} MW'.format(requirement(y_true, y_pred)))
    print('Closeness: {} MW'.format(closeness(y_true, y_pred)))
    print('Exceeding: {} MW'.format(exceeding(y_true, y_pred)))

    print('\n')
    print('Max Exceeding: {} MW'.format(max_exceeding(y_true, y_pred)))
    print('Average reserve ramp rate: {} MW/hr'.format(reserve_ramp_rate(y_true, y_pred)))
