# ==== TODO: Consider having option to disallow overlaps between subsequent training samples

# ==== Change log v1.2 (11/07/2020) ====
# Switched to autogeneration of column names in the M by N matrix based on lag and feature name
# Transposed the M*N matrix in the output. Time is in index and different features are in column
# Move datetime into the index of the input and output dataframe
# Unify input and output processing into one for loop to further streamline the code
# Allow to not include a feature by inputting lag_term_start> lag_term_end
# Streamline the lag term start and end list into a dictionary for further flexibility

# ==== Change log v1.1 (11/01/2020) ====

# Updated to better facilitate future changes in what M would be - both the broad types of predictors and # of lag
# terms corresponding to each
# An additional, important user input is # of lag-terms needed for each predictor
# A function calculates response at each time-point- again to facilitate future changes
# Script now reads in 2 files, 1st containing values and 2nd containing binary flags indicating whether data is valid
# Now, TRUE = Good, FALSE = Bad -> Based on change in data cleaning upstream
# Some efficiency updates

# ====V1.0 originally created (10/22/2020) ====

import os
import numpy as np
import pandas as pd
import utility

# ==== User inputs ====
# the name of the model version that this data would serve
model_name = "rescue_v1_1_no_calendar"

# Define the amount of lag terms that would end up in the input for each feature type
# +1->Forecast time, 0->Present time, -1->1 time step in past, -2->2 time steps in past...
# E.g. 1: start = -2, end = -1 implies only include values from 2 past time steps.
# E.g. 2: start = 0 , end = -1 implies do not include any terms for this feature.
lag_term_start_predictors = np.array([-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2])
lag_term_end_predictors = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
response_lead_term = 1  # As a gentle reminder, its relative to present time, T0. So, 1 implies T0+1

# Those associated with calculating calendar terms - currently solar hour angle and day angle and # of days
# # of Days will account for increasing nameplate, improving forecast accuracy and other phenomena
# that take place over time.
longitude = -119.4179  # Roughly passing through the center of CA
time_difference_from_UTC = -8  # hours. Timestamps for input data are in PST


# ==== Helper functions that don't need user intervention ====
# User needs to define what the response variable is
def calculate_response_variable(raw_data_df):
    """
    Calculates and stores response variable that the ML model will be trained to predict
    :param raw_data_df:
    :return: response_values_df - carrying the same data format as raw_data_df but
    with the response variable calculated
    """

    # Fill it in per user provided definition of the response variable
    # NOTE: When a response is calculated using a predictor that was pd.NA, you want the response
    # to be pd.NA as well. Thus, be careful with using functions like pd.sum() which yield sum
    # of NAs to be = 0 for example.
    net_load_rtpd_forecast = raw_data_df["Load_RTPD_Forecast"] - \
                             raw_data_df["Solar_RTPD_Forecast"] - \
                             raw_data_df["Wind_RTPD_Forecast"]
    net_load_rtd_1_forecast = raw_data_df["Load_RTD_1_Forecast"] - \
                              raw_data_df["Solar_RTD_1_Forecast"] - \
                              raw_data_df["Wind_RTD_1_Forecast"]
    net_load_rtd_2_forecast = raw_data_df["Load_RTD_2_Forecast"] - \
                              raw_data_df["Solar_RTD_2_Forecast"] - \
                              raw_data_df["Wind_RTD_2_Forecast"]
    net_load_rtd_3_forecast = raw_data_df["Load_RTD_3_Forecast"] - \
                              raw_data_df["Solar_RTD_3_Forecast"] - \
                              raw_data_df["Wind_RTD_3_Forecast"]

    response_values_df = net_load_rtpd_forecast - (net_load_rtd_1_forecast + net_load_rtd_2_forecast + \
                                                   net_load_rtd_3_forecast) / 3.0

    return response_values_df


def calculate_calendar_based_predictors(datetime_arr, longitude, time_difference_from_UTC, start_date=None):
    """
    Calculated calendar-based inputs at each time point in the trainval set for ML model. Currently includes solar hour,
    day angle and # of days passed since a start-date which can either be a user input or the first day in the trainval
    dataset.

    Inputs:
    datetime_arr(pd.DatetimeIndex)
    longitude(float): Longitude to be used to calculate local solar time in degrees. East->postive, West->Negative
    time_difference_from_from_UTC(int/float): Time-difference (in hours) between local time and
    Universal Coordinated TIme (UTC)
    start_date(DateTime) = Unless user-specified, is set to first entry in datetime_arr

    Output:
    solar_hour_angle_arr (Array of floats): Hour angle in degrees for each timepoint in datetime_arr
    solar_day_angle_arr (Array of floats): Day angle in degrees for each timepoint in datetime_arr
    days_from_start_date_arr (Array of ints): Days passed since a particular start date, defined for each timepoint in datetime_arr

    Reference for formulae:https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time
    """
    # Steps leading up to calculation of local solar time
    day_of_year_arr = datetime_arr.dayofyear
    # Equation of time (EoT) corrects for eccentricity of earth's orbit and axial tilt
    solar_day_angle_arr = (360 / 365) * (day_of_year_arr - 81)  # degrees
    solar_day_angle_in_radians_arr = np.deg2rad(solar_day_angle_arr)  # radians
    EoT_arr = 9.87 * np.sin(2 * solar_day_angle_in_radians_arr) - 7.53 * np.cos(
        solar_day_angle_in_radians_arr) - 1.5 * np.sin(solar_day_angle_in_radians_arr)  # minutes
    # Time correction sums up time difference due to EoT and longitudinal difference between local time
    # zone and local longitude
    local_std_time_meridian = 15 * time_difference_from_UTC  # degrees
    time_correction_arr = 4 * (longitude - local_std_time_meridian) + EoT_arr  # minutes
    # Calculate local solar time using local time and time correction calculated above
    local_solar_time_arr = datetime_arr.hour + (datetime_arr.minute / 60) + (time_correction_arr / 60)  # hours
    # Calculate solar hour angle corresponding to the local solar time
    solar_hour_angle_arr = 15 * (local_solar_time_arr - 12)  # degrees

    # Calculate days passed since start date
    if start_date is None:
        start_date = datetime_arr[0]
    days_from_start_date_arr = (datetime_arr - start_date).days

    return solar_hour_angle_arr, solar_day_angle_arr, days_from_start_date_arr


def pad_raw_data_w_lag_lead(raw_data_df, lag_term_start_predictors, lag_term_end_predictors,
                            response_lead_term):
    '''
    A function to pad the raw data files in both the lag (backwards) and the lead (forwards) direction
    As the lag terms used in input downstream make use of vectorized calculation. A uniform padding allows
    easier manipulation of the data and constant dataframe size. 
    
    Input: 
    raw_data_df: original raw data dataframe
    lag_term_start_predictors: the start of the lag terms for the predictors/features
    lag_term_end_predictors: the end of the lag terms used as the predictors/features 
    response_lead_term: the amount of lead time (expressed in interval) for the response variable
    
    Output:
    raw_data_df: The origianal raw data dataframe padded with enough NaNs in lag and lead direction
    raw_data_start_idx: in the now padded dataframe, where does the raw data actually start
    raw_data_end_idx: in the now padded dataframe, where does the raw data actually ends
    '''

    # Calculate the maximum amount of lag and lead to determine length of padding
    raw_data_start_time, raw_data_end_time = raw_data_df.index[0], raw_data_df.index[-1]
    max_num_lag_terms = min([lag_term_start_predictors.min(), 0])
    max_num_lead_terms = max([lag_term_end_predictors.max(), 0, response_lead_term])

    # Discern the raw data's inherent frequency. If it's inconsistent then all is moot.
    raw_data_freq = pd.infer_freq(raw_data_df.index)
    assert raw_data_freq is not None, "Raw data does not have equally spaced index! Cannot discern Frequency!"
    # Create padding for lag and lead terms
    lag_terms_timeshift = np.arange(max_num_lag_terms, 0) * pd.Timedelta(raw_data_freq)
    lead_terms_timeshift = np.arange(1, max_num_lead_terms + 1) * pd.Timedelta(raw_data_freq)
    raw_data_lag_pad = pd.DataFrame(index=raw_data_start_time + lag_terms_timeshift, columns=raw_data_df.columns)
    raw_data_lead_pad = pd.DataFrame(index=raw_data_end_time + lead_terms_timeshift, columns=raw_data_df.columns)

    # Append the padding to the raw data frame
    raw_data_df = pd.concat([raw_data_lag_pad, raw_data_df, raw_data_lead_pad])
    # calculate the start and end idx of the raw_data in the padded dataframe
    raw_data_start_idx, raw_data_end_idx = -max_num_lag_terms, raw_data_df.shape[0] - max_num_lead_terms

    return raw_data_df, raw_data_start_idx, raw_data_end_idx


def main(model_name=model_name, lag_term_start_predictors=lag_term_start_predictors,
         lag_term_end_predictors=lag_term_end_predictors,response_lead_term = response_lead_term,
         longitude = longitude, time_difference_from_UTC = time_difference_from_UTC):

    # ==== Constants for use in script that DON'T need to be user defined ====
    # Labels for response (output model is trained to predict)
    response_col_name = "Net_Load_Forecast_Error"
    # The names of several calendar related terms
    hour_angle_col_name = "Hour_Angle"
    day_angle_col_name = "Day_Angle"
    days_from_start_date_col_name = "Days_from_Start_Date"


    # ==== 1. Reading in raw data and validate/modify the data for downstream manipulation ====
    # Paths to read raw data files from and to store outputs in. Defined in the dir_structure class in utility
    dir_str = utility.Dir_Structure(model_name=model_name)

    # Read in raw data to be used to create predictors and response variables
    raw_data_df = pd.read_csv(dir_str.raw_data_path, index_col=0, parse_dates=True, infer_datetime_format=True)
    raw_data_validity = pd.read_csv(dir_str.raw_data_validity_path, index_col=0, parse_dates=True,
                                    infer_datetime_format=True)

    # Check the validity mask of the raw data is consistent with data's shape
    assert (raw_data_df.index == raw_data_validity.index).all(), "Validity mask and Data index inconsistent!"
    assert (raw_data_df.columns == raw_data_validity.columns).all(), "Validity mask and Data fields inconsistent!"

    # Embed info about validity into df holding values so we can use the latter alone going forward
    raw_data_df[~raw_data_validity] = None
    num_time_points = raw_data_df.shape[0]

    # Pad the raw data with NaNs in both the lag and lead direction for downstream data manipulation
    raw_data_df, raw_data_start_idx, raw_data_end_idx = pad_raw_data_w_lag_lead(raw_data_df, lag_term_start_predictors,
                                                                                lag_term_end_predictors,
                                                                                response_lead_term)
    raw_data_start_date = raw_data_df.index[raw_data_start_idx]

    # ==== 2. Add in calendar terms for the raw data ====
    print("Calculating calendar-based predictors....")
    raw_data_df[hour_angle_col_name], raw_data_df[day_angle_col_name], raw_data_df[days_from_start_date_col_name] = \
        calculate_calendar_based_predictors(raw_data_df.index, longitude, time_difference_from_UTC, raw_data_start_date)

    # ==== 3. Add in netload forecast difference for the raw data ====
    print("Calculating response....")
    raw_data_df[response_col_name] = calculate_response_variable(raw_data_df)

    # Revise the lag term array since we are extending the original data
    num_feature_ext = len(raw_data_df.columns) - len(lag_term_start_predictors)
    lag_term_start_predictors = np.hstack(
        (lag_term_start_predictors, np.ones(num_feature_ext, dtype=int) * response_lead_term))
    lag_term_end_predictors = np.hstack(
        (lag_term_end_predictors, np.ones(num_feature_ext, dtype=int) * response_lead_term))

    # ==== 4. Using vectorized operations to construct lag terms ====
    print("Creating trainval samples for all time-points ....")
    # Initialize collectors to hold (and later save) trainval data in
    trainval_data_df = pd.DataFrame(None, index=raw_data_df.index[raw_data_start_idx:raw_data_end_idx])

    # Colelct lag term predictors for all trainval samples
    # Iterate over each type of lag term predictor
    for feature_idx, feature_type in enumerate(raw_data_df.columns):

        # obtain lag term start and end offset for a certain feature
        lag_term_start = lag_term_start_predictors[feature_idx]
        lag_term_end = lag_term_end_predictors[feature_idx]

        # Iterate over each time step for current predictor type
        for time_step in range(lag_term_start, lag_term_end + 1):
            label = "{}_T{:+}".format(feature_type, time_step)
            trainval_data_df[label] = (raw_data_df[feature_type]
                                       .iloc[raw_data_start_idx + time_step: raw_data_end_idx + time_step].values)

    # ==== 5. Drop invalid terms and store to hard drive ====
    # Identify trainval samples wherein all lag term of features and responses are valid
    # If any entry is pd.NA, it is invalid
    print("{} of {} trainval samples are valid"
          .format(trainval_data_df.notna().all(axis=1).sum(), trainval_data_df.shape[0]))
    print("Proceeding to delete the rest....")
    trainval_data_df = trainval_data_df.dropna()

    # Only retain trainval samples wherein predictor(s) and response are both valid
    output_idx = np.where([response_col_name in col for col in trainval_data_df.columns])[0][0]
    trainval_outputs_df = trainval_data_df.pop(trainval_data_df.columns[output_idx])

    # Save trainval samples
    print("Saving files......")
    trainval_data_df.to_pickle(dir_str.input_trainval_path)
    trainval_outputs_df.to_pickle(dir_str.output_trainval_path)
    print("All done!")


# run as a script
if __name__ == '__main__':
    main()
