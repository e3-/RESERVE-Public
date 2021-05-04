# ==== TODO: Consider having option to disallow overlaps between subsequent training samples
# ==== Change log v1.4 (4/28/2021) ====
# Modified script to generate trainval inputs and outputs for 5 min prediction
# Introduced a variable named lag_term_step_predictors to ensure we can sample every third RTPD term and every RTD term
# while creating 5 min data. Also added a new RTD_Interval_ID predictor which is 0/1 or 2 based on its position within
# a RTPD interval

# ==== Change log v1.3 (1/13/2021) ====
# Enabled script' ability to support both single and multi-objective learning
# if the multi_obj_learning_flag is set to True, response variables will be net load, load, solar and wind forecast
# errors in that order. Else, the sole response variable will be the net load forecast error

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
model_name = "rescue_5_min_v1_single_obj"

# Define the amount of lag terms that would end up in the input for each feature type
# +1->Forecast time, 0->Present time, -1->1 time step in past, -2->2 time steps in past...
# E.g. 1: start = -2, end = -1 implies only include values from 2 past time steps.
# E.g. 2: start = 0 , end = -1 implies do not include any terms for this feature.
lag_term_start_predictors = np.array([-6, -6, -6, -6, -6, -6])
lag_term_end_predictors = np.array([-1, 3, -1, 3, -1, 3])
# Step size between subsequent lag terms for ML model. If 2, implies pick every 2nd lag term between start to end
# defined above.
# Use case-> When a 15-min predictor is just repeated thrice to get a 5-min predictor, you can pick every 3rd value
# in that time-series to avoid redundancy
lag_term_step_predictors = np.array([1, 3, 1, 3, 1, 3])
# Currently, the same lead term will be applicable to each response variable, if we have several of 'em
response_lead_term = 3  # As a gentle reminder, its relative to present time, T0. So, 1 implies T0+1 for eg

# Those associated with calculating calendar terms - currently solar hour angle and day angle and # of days
# # of Days will account for increasing nameplate, improving forecast accuracy and other phenomena
# that take place over time.
longitude = -119.4179  # Roughly passing through the center of CA
time_difference_from_UTC = -8  # hours. Timestamps for input data are in PST
rtpd_interval = 15 # minutes
rtd_interval = 5 # minutes

# This flag should be set to True, if you want model response to be all of net load, load, solar and wind forecast
# errors in that order. Set to False for creating a single response variable, the net load forecast error
multi_obj_learning_flag = False

# ==== Helper functions that don't need user intervention ====
# User needs to define what the response variable is
def calculate_response_variables(raw_data_df, response_col_names):
    """
    Calculates and stores response variable(s) that the ML model will be trained to predict
    :param raw_data_df: Df containing all predictors. Some if not all of them will be used to calculate response(s)
    :param response_col_names: List with column names corresponding to response variables to be calculated in this
                               function
    :return: response_values_df - carrying the same data format as raw_data_df but
    with the response variable(s) calculated
    """
    # Initialize df to store response variable(s)
    response_values_df = pd.DataFrame(index=raw_data_df.index, columns=response_col_names)
    # Fill it in per user provided definition of the response variable(s)
    # NOTE: When a response is calculated using a predictor that was pd.NA, you want the response
    # to be pd.NA as well. Thus, be careful with using functions like pd.sum() which yield sum
    # of NAs to be = 0 for example.

    load_forecast_error = raw_data_df["Load_RTPD_Forecast"] - raw_data_df["Load_RTD_Forecast"]

    solar_forecast_error = raw_data_df["Solar_RTPD_Forecast"] - raw_data_df["Solar_RTD_Forecast"]

    wind_forecast_error = raw_data_df["Wind_RTPD_Forecast"] - raw_data_df["Wind_RTD_Forecast"]

    net_load_forecast_error = load_forecast_error - solar_forecast_error - wind_forecast_error

    # Net load forecast error will be the sole response in single objective learning
    response_values_df.loc[:, "Net_Load_Forecast_Error"] = net_load_forecast_error

    # The response variable(s) below have been added for multi-objective learning
    if len(response_col_names) > 1:
        response_values_df.loc[:, "Load_Forecast_Error"] = load_forecast_error
        response_values_df.loc[:, "Solar_Forecast_Error"] = solar_forecast_error
        response_values_df.loc[:, "Wind_Forecast_Error"] = wind_forecast_error

    return response_values_df


def calculate_calendar_based_predictors(datetime_arr, longitude, time_difference_from_UTC,
                                        rtpd_interval, rtd_interval,
                                        start_date=None):
    """
    Calculated calendar-based inputs at each time point in the trainval set for ML model. Currently includes solar hour,
    day angle and # of days passed since a start-date which can either be a user input or the first day in the trainval
    dataset.

    Inputs:
    datetime_arr(pd.DatetimeIndex)
    longitude(float): Longitude to be used to calculate local solar time in degrees. East->postive, West->Negative
    time_difference_from_from_UTC(int/float): Time-difference (in hours) between local time and
    Universal Coordinated TIme (UTC)
    rtpd_interval(int/float) - Length of rtpd interval in min
    rtd_interval(int/float) - Length of rtd interval in min. Will be used to assign (sub)interval ids for each rtd
    interval comprised within a rtpd interval
    start_date(DateTime) = Unless user-specified, is set to first entry in datetime_arr

    Output:
    solar_hour_angle_arr (Array of floats): Hour angle in degrees for each timepoint in datetime_arr
    solar_day_angle_arr (Array of floats): Day angle in degrees for each timepoint in datetime_arr
    days_from_start_date_arr (Array of ints): Days passed since a particular start date, defined for each timepoint in datetime_arr
    interval_id_arr (Array of ints): Equals 0, 1, 2 for the 3 RTD intervals comprised within each RTPD interval

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

    # Calculate rtd interval ids w.r.t each rtpd interval
    # For eg, rtd interval starting 12:00 = 0, 12:05 = 1, 12:10 = 2, if rtpd interval spans from 12:00 to 12:15
    interval_id_arr = np.zeros_like(datetime_arr, dtype=int)
    for interval_id in range(int(rtpd_interval/rtd_interval)):
        condition_arr = datetime_arr.minute % rtpd_interval == rtd_interval * interval_id
        interval_id_arr[condition_arr] = interval_id

    return solar_hour_angle_arr, solar_day_angle_arr, days_from_start_date_arr, interval_id_arr


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
         lag_term_end_predictors=lag_term_end_predictors, lag_term_step_predictors=lag_term_step_predictors,
         response_lead_term=response_lead_term,
         longitude=longitude, time_difference_from_UTC=time_difference_from_UTC,
         rtpd_interval=rtpd_interval, rtd_interval=rtd_interval,
         multi_obj_learning_flag=multi_obj_learning_flag):
    # ==== Constants for use in script that DON'T need to be user defined ====
    # Labels for response (output(s) model is trained to predict)
    if multi_obj_learning_flag:
        # You can change these labels, but the order MUST be net load->load->solar->wind
        # To change order or add/remove any response variables, you will need to change the function
        # calculate_response_variables too
        response_col_names = ["Net_Load_Forecast_Error", "Load_Forecast_Error", "Solar_Forecast_Error",
                              "Wind_Forecast_Error"]
    else:
        response_col_names = ["Net_Load_Forecast_Error"]

    # The names of several calendar related terms
    hour_angle_col_name = "Hour_Angle"
    day_angle_col_name = "Day_Angle"
    days_from_start_date_col_name = "Days_from_Start_Date"
    interval_id_col_name = "5_Min_Interval_ID"

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

    # Pad the raw data with NaNs in both the lag and lead direction for downstream data manipulation
    raw_data_df, raw_data_start_idx, raw_data_end_idx = pad_raw_data_w_lag_lead(raw_data_df, lag_term_start_predictors,
                                                                                lag_term_end_predictors,
                                                                                response_lead_term)
    raw_data_start_date = raw_data_df.index[raw_data_start_idx]

    # ==== 2. Add in calendar terms for the raw data ====
    print("Calculating calendar-based predictors....")
    raw_data_df[hour_angle_col_name], raw_data_df[day_angle_col_name], raw_data_df[days_from_start_date_col_name], raw_data_df[interval_id_col_name] = \
        calculate_calendar_based_predictors(raw_data_df.index, longitude, time_difference_from_UTC,
                                            rtpd_interval, rtd_interval, raw_data_start_date)

    # ==== 3. Add in net-load forecast difference and load, solar, wind (if multi-obj) forecast difference
    # for the raw data. These will be used as response variable(s) ====
    print("Calculating response(s)....")
    response_df = calculate_response_variables(raw_data_df, response_col_names)
    raw_data_df = pd.concat([raw_data_df, response_df], axis=1)

    # Revise the lag term array since we are extending the original data
    num_feature_ext = len(raw_data_df.columns) - len(lag_term_start_predictors)
    lag_term_start_predictors = np.hstack(
        (lag_term_start_predictors, np.ones(num_feature_ext, dtype=int) * response_lead_term))
    lag_term_end_predictors = np.hstack(
        (lag_term_end_predictors, np.ones(num_feature_ext, dtype=int) * response_lead_term))
    # We are going from T0 to prediction time in 1 step
    lag_term_step_predictors = np.hstack(
        (lag_term_step_predictors, np.ones(num_feature_ext, dtype=int) * 1))

    # ==== 4. Using vectorized operations to construct lag terms ====
    print("Creating trainval samples for all time-points ....")
    # Initialize collectors to hold (and later save) trainval data in
    trainval_data_df = pd.DataFrame(None, index=raw_data_df.index[raw_data_start_idx:raw_data_end_idx])

    # Collect lag term predictors for all trainval samples
    # Iterate over each type of lag term predictor
    for feature_idx, feature_type in enumerate(raw_data_df.columns):

        # obtain lag term start and end offset, as well as step size for a certain feature
        lag_term_start = lag_term_start_predictors[feature_idx]
        lag_term_end = lag_term_end_predictors[feature_idx]
        lag_term_step = lag_term_step_predictors[feature_idx]

        # Iterate over each time step for current predictor type
        for time_step in range(lag_term_start, lag_term_end + 1, lag_term_step):
            label = "{}_T{:+}".format(feature_type, time_step)
            trainval_data_df[label] = (raw_data_df[feature_type]
                                       .iloc[raw_data_start_idx + time_step: raw_data_end_idx + time_step].values)

    # ==== 5. Drop invalid terms and store to hard drive ====
    # Identify trainval samples wherein all lag term of features and responses are valid
    # If any entry is pd.NA, it is invalid
    print("{} of {} trainval samples are valid"
          .format(trainval_data_df.notna().all(axis=1).sum(), trainval_data_df.shape[0]))
    print("Proceeding to delete the rest....")
    # Only retain trainval samples wherein predictor(s) and response(s) are both valid
    trainval_data_df = trainval_data_df.dropna()

    # Separate predictors (model inputs) from response (model output(s))
    response_col_names = ["{}_T{:+}".format(name,response_lead_term) for name in response_col_names ]
    trainval_outputs_df = trainval_data_df.loc[:, response_col_names].copy()
    trainval_data_df = trainval_data_df.drop(columns=trainval_outputs_df.columns)

    # Save trainval samples
    print("Saving files......")
    trainval_data_df.to_pickle(dir_str.input_trainval_path)
    trainval_outputs_df.to_pickle(dir_str.output_trainval_path)
    print("All done!")


# run as a script
if __name__ == '__main__':
    main()
