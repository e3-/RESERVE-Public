# ==== TODO: Consider having option to disallow overlaps between subsequent training samples
# ==== Change log v1.5 (5/29/2021) ====
# Added functionality to take raw outputs from the data-checker, ensure each of them match desired temporal
# resolution and then create the trainval inputs and outputs as before

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

# Declare files that data-checker produces but aren't to be used to create inputs for ML
files_to_ignore = ["summary_all_files.csv", "archive"]

# Temporal characteristics required for making trainval set
start_date = "01-01-2017"  # Inclusive
end_date = "01-01-2020"  # Exclusive
ML_time_step = "5T"  # T implies minutes

# Column names corresponding to those in data-checker output files
col_name_for_value = "Forecast_Interval_Avg_MW"
col_name_for_validity_flag = "valid_all_checks"
datetime_col_name = "Datetime_Interval_Start"

# the name of the model version that this data would serve
model_name = "rescue_5_min_v1_single_obj_test"

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
rtpd_interval = 15  # minutes
rtd_interval = 5  # minutes

# This flag should be set to True, if you want model response to be all of net load, load, solar and wind forecast
# errors in that order. Set to False for creating a single response variable, the net load forecast error
multi_obj_learning_flag = False

# ==== Helper functions that don't need user intervention ====


def merge_df(df_numeric_1, df_numeric_2, df_bool_1, df_bool_2):
    """
    Perform outer merging of dataframes
    :param df_numeric_1:DataFrame holding numeric values
    :param df_numeric_2:DataFrame holding numeric values, to be merged with df_1
    :param df_bool_1:DataFrame holding booleans that highlight validity of df_numeric_1
    :param df_bool_2:DataFrame holding booleans that highlight validity of df_numeric_2, to be merged with df_1
    :return: df_numeric_merged, df_bool_merged: DataFrames resulting from outer merge operation
    """
    df_numeric_merged = df_numeric_1.merge(
        df_numeric_2, right_index=True, left_index=True, how="outer"
    )
    df_bool_merged = df_bool_1.merge(
        df_bool_2, right_index=True, left_index=True, how="outer"
    )
    # In case index of the two dfs didn't match (can happen if some feature has a longer/shorter historical
    # record), fill the resulting nans with 0 and FALSE resp. Numeric data will be used in conjunction with bool data
    # later. FALSE data will be effectively ignored. So, we don't need more sophisticated replacements for nans
    df_numeric_merged = df_numeric_merged.fillna(0.0)
    df_bool_merged = df_bool_merged.fillna(False)
    return df_numeric_merged, df_bool_merged


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
    response_values_df = pd.DataFrame(
        index=raw_data_df.index, columns=response_col_names
    )
    # Fill it in per user provided definition of the response variable(s)
    # NOTE: When a response is calculated using a predictor that was pd.NA, you want the response
    # to be pd.NA as well. Thus, be careful with using functions like pd.sum() which yield sum
    # of NAs to be = 0 for example.

    load_forecast_error = (
        raw_data_df["Load_RTPD_Forecast"] - raw_data_df["Load_RTD_Forecast"]
    )

    solar_forecast_error = (
        raw_data_df["Solar_RTPD_Forecast"] - raw_data_df["Solar_RTD_Forecast"]
    )

    wind_forecast_error = (
        raw_data_df["Wind_RTPD_Forecast"] - raw_data_df["Wind_RTD_Forecast"]
    )

    net_load_forecast_error = (
        load_forecast_error - solar_forecast_error - wind_forecast_error
    )

    # Net load forecast error will be the sole response in single objective learning
    response_values_df.loc[:, "Net_Load_Forecast_Error"] = net_load_forecast_error

    # The response variable(s) below have been added for multi-objective learning
    if len(response_col_names) > 1:
        response_values_df.loc[:, "Load_Forecast_Error"] = load_forecast_error
        response_values_df.loc[:, "Solar_Forecast_Error"] = solar_forecast_error
        response_values_df.loc[:, "Wind_Forecast_Error"] = wind_forecast_error

    return response_values_df


def calculate_calendar_based_predictors(
    datetime_arr,
    longitude,
    time_difference_from_UTC,
    rtpd_interval,
    rtd_interval,
    start_date=None,
):
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
    EoT_arr = (
        9.87 * np.sin(2 * solar_day_angle_in_radians_arr)
        - 7.53 * np.cos(solar_day_angle_in_radians_arr)
        - 1.5 * np.sin(solar_day_angle_in_radians_arr)
    )  # minutes
    # Time correction sums up time difference due to EoT and longitudinal difference between local time
    # zone and local longitude
    local_std_time_meridian = 15 * time_difference_from_UTC  # degrees
    time_correction_arr = 4 * (longitude - local_std_time_meridian) + EoT_arr  # minutes
    # Calculate local solar time using local time and time correction calculated above
    local_solar_time_arr = (
        datetime_arr.hour + (datetime_arr.minute / 60) + (time_correction_arr / 60)
    )  # hours
    # Calculate solar hour angle corresponding to the local solar time
    solar_hour_angle_arr = 15 * (local_solar_time_arr - 12)  # degrees

    # Calculate days passed since start date
    if start_date is None:
        start_date = datetime_arr[0]
    days_from_start_date_arr = (datetime_arr - start_date).days

    # Calculate rtd interval ids w.r.t each rtpd interval
    # For eg, rtd interval starting 12:00 = 0, 12:05 = 1, 12:10 = 2, if rtpd interval spans from 12:00 to 12:15
    interval_id_arr = np.zeros_like(datetime_arr, dtype=int)
    for interval_id in range(int(rtpd_interval / rtd_interval)):
        condition_arr = (
            datetime_arr.minute % rtpd_interval == rtd_interval * interval_id
        )
        interval_id_arr[condition_arr] = interval_id

    return (
        solar_hour_angle_arr,
        solar_day_angle_arr,
        days_from_start_date_arr,
        interval_id_arr,
    )


def pad_raw_data_w_lag_lead(
    raw_data_df, lag_term_start_predictors, lag_term_end_predictors, response_lead_term
):
    """
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
    """

    # Calculate the maximum amount of lag and lead to determine length of padding
    raw_data_start_time, raw_data_end_time = raw_data_df.index[0], raw_data_df.index[-1]
    max_num_lag_terms = min([lag_term_start_predictors.min(), 0])
    max_num_lead_terms = max([lag_term_end_predictors.max(), 0, response_lead_term])

    # Discern the raw data's inherent frequency. If it's inconsistent then all is moot.
    raw_data_freq = pd.infer_freq(raw_data_df.index)
    assert (
        raw_data_freq is not None
    ), "Raw data does not have equally spaced index! Cannot discern Frequency!"
    # Create padding for lag and lead terms
    lag_terms_timeshift = np.arange(max_num_lag_terms, 0) * pd.Timedelta(raw_data_freq)
    lead_terms_timeshift = np.arange(1, max_num_lead_terms + 1) * pd.Timedelta(
        raw_data_freq
    )
    raw_data_lag_pad = pd.DataFrame(
        index=raw_data_start_time + lag_terms_timeshift, columns=raw_data_df.columns
    )
    raw_data_lead_pad = pd.DataFrame(
        index=raw_data_end_time + lead_terms_timeshift, columns=raw_data_df.columns
    )

    # Append the padding to the raw data frame
    raw_data_df = pd.concat([raw_data_lag_pad, raw_data_df, raw_data_lead_pad])
    # calculate the start and end idx of the raw_data in the padded dataframe
    raw_data_start_idx, raw_data_end_idx = (
        -max_num_lag_terms,
        raw_data_df.shape[0] - max_num_lead_terms,
    )

    return raw_data_df, raw_data_start_idx, raw_data_end_idx


def main(
    files_to_ignore=files_to_ignore,
    start_date=start_date,
    end_date=end_date,
    ML_time_step=ML_time_step,
    model_name=model_name,
    lag_term_start_predictors=lag_term_start_predictors,
    lag_term_end_predictors=lag_term_end_predictors,
    lag_term_step_predictors=lag_term_step_predictors,
    response_lead_term=response_lead_term,
    longitude=longitude,
    time_difference_from_UTC=time_difference_from_UTC,
    rtpd_interval=rtpd_interval,
    rtd_interval=rtd_interval,
    multi_obj_learning_flag=multi_obj_learning_flag,
):
    # ==== Constants for use in script that DON'T need to be user defined ====

    # Column names corresponding to those in data-checker output files
    col_name_for_value = "Forecast_Interval_Avg_MW"
    col_name_for_validity_flag = "valid_all_checks"
    datetime_col_name = "Datetime_Interval_Start"

    # Labels for response (output(s) model is trained to predict)
    if multi_obj_learning_flag:
        # You can change these labels, but the order MUST be net load->load->solar->wind
        # To change order or add/remove any response variables, you will need to change the function
        # calculate_response_variables too
        response_col_names = [
            "Net_Load_Forecast_Error",
            "Load_Forecast_Error",
            "Solar_Forecast_Error",
            "Wind_Forecast_Error",
        ]
    else:
        response_col_names = ["Net_Load_Forecast_Error"]

    # The names of several calendar related terms
    hour_angle_col_name = "Hour_Angle"
    day_angle_col_name = "Day_Angle"
    days_from_start_date_col_name = "Days_from_Start_Date"
    interval_id_col_name = "5_Min_Interval_ID"

    # ==== 0. Read in each time-series feature, output from the data-checker and ensure it matches ML time-step ====
    # Paths to read raw data files from and to store outputs in. Defined in the dir_structure class in utility
    dir_str = utility.Dir_Structure(model_name=model_name)
    path_to_data_checker_outputs = dir_str.data_checker_dir

    # Initialize df to hold collated data for ML model
    raw_data_df = pd.DataFrame(
        index=pd.date_range(
            start=start_date, end=end_date, freq=ML_time_step, closed="left"
        )
    )
    raw_data_validity = raw_data_df.copy()

    # Iterate over each feature output from data-checker
    print(
        "Reading in outputs from data-checker and ensuring temporal resolution meets requirement for model...."
    )
    for file in os.listdir(path_to_data_checker_outputs):
        feature_name = file.strip(".csv")
        if file in files_to_ignore:
            print(
                "Skipping {}. Not assumed to be holding inputs for ML model".format(
                    file
                )
            )
            continue
        feature_df = pd.read_csv(
            os.path.join(path_to_data_checker_outputs, file),
            usecols=[datetime_col_name, col_name_for_value, col_name_for_validity_flag],
            index_col=datetime_col_name,
            parse_dates=True,
        )
        # Determine time-step size in the data for this feature
        inferred_freq = pd.infer_freq(feature_df.index)
        # If inferred frequency is 1 of something, say 1 min or 1 H, it will just be represented
        # as "T" or "H". Need to turn it into "1T" or "1H" to interpret as a number using the Timedelta function
        if inferred_freq[0].isdigit():
            feature_time_step = pd.Timedelta(inferred_freq).seconds
        else:
            feature_time_step = pd.Timedelta("1" + inferred_freq).seconds
        # Similarly, determine time-step required for ML model
        inferred_freq = pd.infer_freq(raw_data_df.index)
        if inferred_freq[0].isdigit():
            ML_inputs_time_step = pd.Timedelta(inferred_freq).seconds
        else:
            ML_inputs_time_step = pd.Timedelta("1" + inferred_freq).seconds

        # ==== Option 1 of 3 ====
        # If the feature time-step size matches that needed for ML inputs, place the feature into the ML inputs df
        # without any changes
        if feature_time_step == ML_inputs_time_step:
            raw_data_df, raw_data_validity = merge_df(
                raw_data_df,
                feature_df[col_name_for_value],
                raw_data_validity,
                feature_df[col_name_for_validity_flag],
            )
            # Rename from generic col name to feature-specific name
            raw_data_df = raw_data_df.rename(columns={col_name_for_value: feature_name})
            raw_data_validity = raw_data_validity.rename(
                columns={col_name_for_validity_flag: feature_name}
            )
        # ==== Option 2 of 3 ====
        # If the time step is shorter than that desired, create multiple features
        # Lets say we have 5-min features and ML inputs are on a 15-min resolution. Then, create three 15-minutely features
        # out of the one 5-min feature currently being assessed
        elif feature_time_step < ML_inputs_time_step:
            assert ML_inputs_time_step % feature_time_step == 0, (
                "ML time-step needs to be equal to or a perfect multiple of "
                "feature time-step. Not the case for {}".format(feature_name)
            )
            num_substeps = int(ML_inputs_time_step / feature_time_step)
            # Create new index and labels for the additional features to be created
            feature_name_substep_idx = [
                feature_name + "_" + str(i) for i in range(num_substeps)
            ]
            feature_name_substep_idx = np.tile(
                feature_name_substep_idx, int(feature_df.shape[0] / num_substeps)
            )
            timestamp_idx = np.repeat(feature_df.index[::num_substeps], num_substeps)
            # These will first be turned into a multi-level index for the feature_df and then be "unstacked" to have the
            # timestamps be the sole index and have the unique entries in feature_name_substep_idx become the columns
            feature_df.index = pd.MultiIndex.from_arrays(
                [timestamp_idx, feature_name_substep_idx]
            )
            feature_df = feature_df.unstack()
            raw_data_df, raw_data_validity = merge_df(
                raw_data_df,
                feature_df[col_name_for_value],
                raw_data_validity,
                feature_df[col_name_for_validity_flag],
            )
        # ==== Option 3 of 3 ====
        # If feature time-step is longer than desired, replicate feature
        # For eg, if ML time step is 15 min and feature df has values on hourly resolution, we will repeat the feature
        # value 4 times, once for each 15 min interval in the given hour
        else:
            assert feature_time_step % ML_inputs_time_step == 0, (
                "Feature time-step needs to be equal to or a perfect "
                "multiple of the ML time-step. Not the case for "
                "{}".format(feature_name)
            )
            repeated_feature_df = feature_df.resample(
                str(ML_inputs_time_step) + "S"
            ).pad()
            raw_data_df, raw_data_validity = merge_df(
                raw_data_df,
                repeated_feature_df[col_name_for_value],
                raw_data_validity,
                repeated_feature_df[col_name_for_validity_flag],
            )
            # Rename from generic col name to feature-specific name
            raw_data_df = raw_data_df.rename(columns={col_name_for_value: feature_name})
            raw_data_validity = raw_data_validity.rename(
                columns={col_name_for_validity_flag: feature_name}
            )

    # ==== 1. Identify valid raw data and pad for downstream manipulation ====

    # Check the validity mask of the raw data is consistent with data's shape
    assert (
        raw_data_df.index == raw_data_validity.index
    ).all(), "Validity mask and Data index inconsistent!"
    assert (
        raw_data_df.columns == raw_data_validity.columns
    ).all(), "Validity mask and Data fields inconsistent!"

    # Embed info about validity into df holding values so we can use the latter alone going forward
    raw_data_df[~raw_data_validity] = None

    # Pad the raw data with NaNs in both the lag and lead direction for downstream data manipulation
    raw_data_df, raw_data_start_idx, raw_data_end_idx = pad_raw_data_w_lag_lead(
        raw_data_df,
        lag_term_start_predictors,
        lag_term_end_predictors,
        response_lead_term,
    )
    raw_data_start_date = raw_data_df.index[raw_data_start_idx]

    # ==== 2. Add in calendar terms for the raw data ====
    print("Calculating calendar-based predictors....")
    (
        raw_data_df[hour_angle_col_name],
        raw_data_df[day_angle_col_name],
        raw_data_df[days_from_start_date_col_name],
        raw_data_df[interval_id_col_name],
    ) = calculate_calendar_based_predictors(
        raw_data_df.index,
        longitude,
        time_difference_from_UTC,
        rtpd_interval,
        rtd_interval,
        raw_data_start_date,
    )

    # ==== 3. Add in net-load forecast difference and load, solar, wind (if multi-obj) forecast difference
    # for the raw data. These will be used as response variable(s) ====
    print("Calculating response(s)....")
    response_df = calculate_response_variables(raw_data_df, response_col_names)
    raw_data_df = pd.concat([raw_data_df, response_df], axis=1)

    # Revise the lag term array since we are extending the original data
    num_feature_ext = len(raw_data_df.columns) - len(lag_term_start_predictors)
    lag_term_start_predictors = np.hstack(
        (
            lag_term_start_predictors,
            np.ones(num_feature_ext, dtype=int) * response_lead_term,
        )
    )
    lag_term_end_predictors = np.hstack(
        (
            lag_term_end_predictors,
            np.ones(num_feature_ext, dtype=int) * response_lead_term,
        )
    )
    # We are going from T0 to prediction time in 1 step
    lag_term_step_predictors = np.hstack(
        (lag_term_step_predictors, np.ones(num_feature_ext, dtype=int) * 1)
    )

    # ==== 4. Using vectorized operations to construct lag terms ====
    print("Creating trainval samples for all time-points ....")
    # Initialize collectors to hold (and later save) trainval data in
    trainval_data_df = pd.DataFrame(
        None, index=raw_data_df.index[raw_data_start_idx:raw_data_end_idx]
    )

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
            trainval_data_df[label] = (
                raw_data_df[feature_type]
                .iloc[raw_data_start_idx + time_step : raw_data_end_idx + time_step]
                .values
            )

    # ==== 5. Drop invalid terms and store to hard drive ====
    # Identify trainval samples wherein all lag term of features and responses are valid
    # If any entry is pd.NA, it is invalid
    print(
        "{} of {} trainval samples are valid".format(
            trainval_data_df.notna().all(axis=1).sum(), trainval_data_df.shape[0]
        )
    )
    print("Proceeding to delete the rest....")
    # Only retain trainval samples wherein predictor(s) and response(s) are both valid
    trainval_data_df = trainval_data_df.dropna()

    # Separate predictors (model inputs) from response (model output(s))
    response_col_names = [
        "{}_T{:+}".format(name, response_lead_term) for name in response_col_names
    ]
    trainval_outputs_df = trainval_data_df.loc[:, response_col_names].copy()
    trainval_data_df = trainval_data_df.drop(columns=trainval_outputs_df.columns)

    # Save trainval samples
    print("Saving files......")
    trainval_data_df.to_pickle(dir_str.input_trainval_path)
    trainval_outputs_df.to_pickle(dir_str.output_trainval_path)
    print("All done!")


# run as a script
if __name__ == "__main__":
    main()
