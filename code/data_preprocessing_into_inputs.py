# ############################ LICENSE INFORMATION ############################
# This file is part of the E3 RESERVE Model.

# Copyright (C) 2021 Energy and Environmental Economics, Inc.
# For contact information, go to www.ethree.com

# The E3 RESERVE Model is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The E3 RESERVE Model is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with the E3 RESERVE Model (in the file LICENSE.TXT). If not,
# see <http://www.gnu.org/licenses/>.
# #############################################################################

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
import pathlib
dir_str = utility.Dir_Structure()

# ==== User inputs ====

# The only four response labels allowed are Net_Load_Forecast_Error, Load_Forecast_Error, Solar_Forecast_Error,
# Wind_Forecast_Error. They can appear in arbitrary order and can be repeated or omitted. Strings other than these four
# here would result in errors.
response_col_names = {
    "Net Load": "Net_Load_Forecast_Error",
    "Load": "Load_Forecast_Error",
    "Solar": "Solar_Forecast_Error",
    "Wind": "Wind_Forecast_Error",
}

# Column names corresponding to those in data-checker output files
COL_NAME_FOR_VALUE = "Forecast_Interval_Avg_MW"
COL_NAME_FOR_VALIDITY_FLAG = "valid_all_checks"
COL_NAME_FOR_DT = "Datetime_Interval_Start"

# The names of several calendar related terms
COL_NAME_HOUR_ANGLE = "Hour_Angle"
COL_NAME_DAY_ANGLE = "Day_Angle"
COL_NAME_DAYS_IDX = "Days_from_Start_Date"

# Import RESERVE settings and input file settings
df_model_settings = pd.read_excel(dir_str.RESERVE_settings_path, sheet_name="RESERVE Settings", index_col=[0])
df_data_settings = pd.read_excel(dir_str.RESERVE_settings_path, sheet_name="Input Data Settings", index_col=[0])
# Remove ".csv" tags from lag term dataframe indices
df_data_settings.index = [f.strip('.csv') for f in df_data_settings.index]

# The name of the model version that this data would serve
model_name = df_model_settings.loc["MODEL_NAME", "Value"]
# Temporal characteristics required for making trainval set
ML_time_step = pd.Timedelta(str(df_model_settings.loc["ML_TIME_STEP", "Value"]) + "T")  # T implies minutes
# Used to give a consistent day idx sequencing.
ANCHOR_DATE = df_model_settings.loc["ANCHOR_DATE", "Value"]
# Those associated with calculating calendar terms - currently solar hour angle and day angle and # of days
# # of Days will account for increasing nameplate, improving forecast accuracy and other phenomena
# that take place over time.
latitude = df_model_settings.loc["LATITUDE", "Value"]
longitude = df_model_settings.loc["LONGITUDE", "Value"]
time_difference_from_UTC = df_model_settings.loc["TIME_DIFFERENCE_FROM_UTC", "Value"]  # hours. Timestamps for input data are in PST
# Currently, the same lead term will be applicable to each response variable, if we have several of 'em
response_lead_term = df_model_settings.loc["RESPONSE_LEAD_TERM", "Value"]  # As a gentle reminder, its relative to present time, T0. So, 1 implies T0+1 for eg

# Define the amount of lag terms that would end up in the input for each feature type
# +1->Forecast time, 0->Present time, -1->1 time step in past, -2->2 time steps in past...
# E.g. 1: start = -2, end = -1 implies only include values from 2 past time steps.
# E.g. 2: start = 0 , end = -1 implies do not include any terms for this feature.
lag_term_start_predictors = df_data_settings["Lag Term Start Interval"]
lag_term_end_predictors = df_data_settings["Lag Term End Interval"]
# Step size between subsequent lag terms for ML model. If 2, implies pick every 2nd lag term between start to end
# defined above.
# Use case-> When a 15-min predictor is just repeated thrice to get a 5-min predictor, you can pick every 3rd value
# in that time-series to avoid redundancy
lag_term_step_predictors = df_data_settings["Lag Term Interval Step"]


# ==== Helper functions that don't need user intervention ====
# User needs to define what the response variable is
def calculate_response_variables(raw_data_df, response_col_names, df_data_settings):
    """
    Calculates and stores response variable(s) that the ML model will be trained to predict
    :param raw_data_df: Df containing all predictors. Some if not all of them will be used to calculate response(s)
    :param response_col_names: List with column names corresponding to response variables to be calculated in this
                               function
    :param df_data_settings: Data frame with input
    :return: response_values_df - carrying the same data format as raw_data_df but
    with the response variable(s) calculated
    """
    # Initialize df to store response variable(s)
    response_values_df = pd.DataFrame(
        index=raw_data_df.index, columns=response_col_names.values()
    )
    # Fill it in per user provided definition of the response variable(s)
    # NOTE: When a response is calculated using a predictor that was pd.NA, you want the response
    # to be pd.NA as well. Thus, be careful with using functions like pd.sum() which yield sum
    # of NAs to be = 0 for example.

    for key in response_col_names.keys():

        if key == "Net Load":
            # Calculate net load from other response variables once they have been calculated
            continue

        # Define column name of response variable
        col_name = response_col_names[key]

        # Get actual/forecast features corresponding to key
        temp = df_data_settings.loc[df_data_settings["Feature"] == key, "Type"] # Get feature names of actual and forecast
        actual_feature_name = temp.index[temp == "Actual"][0]
        forecast_feature_name = temp.index[temp == "Forecast"][0]

        # Calculate forecast error
        forecast_error = raw_data_df[forecast_feature_name] - raw_data_df[actual_feature_name]

        # Save to response value dataframe
        response_values_df[col_name] = forecast_error

    # Calculate net load in response value dataframe
    response_values_df[response_col_names["Net Load"]] = response_values_df[response_col_names["Load"]] - \
                                                         response_values_df[response_col_names["Wind"]] - \
                                                         response_values_df[response_col_names["Solar"]]

    """
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

    for col_name in response_col_names:

        if "Net_Load_Forecast_Error" == col_name:
            response_values_df[col_name] = net_load_forecast_error
        elif "Load_Forecast_Error" == col_name:
            response_values_df[col_name] = load_forecast_error
        elif "Solar_Forecast_Error" == col_name:
            response_values_df[col_name] = solar_forecast_error
        elif "Wind_Forecast_Error" == col_name:
            response_values_df[col_name] = wind_forecast_error
        else:
            raise ValueError("{} is undefined as a response variable!".format(col_name))
    """

    return response_values_df


def calculate_calendar_based_predictors(
    datetime_arr, longitude, time_difference_from_UTC
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

    Output:
    solar_hour_angle_arr (Array of floats): Hour angle in degrees for each timepoint in datetime_arr
    solar_day_angle_arr (Array of floats): Day angle in degrees for each timepoint in datetime_arr
    days_from_start_date_arr (Array of ints): Days passed since a particular start date, defined for each timepoint in datetime_arr

    Reference for formulae:C.B.Honsberg and S.G.Bowden, “Photovoltaics Education Website,” www.pveducation.org, 2019
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
    days_from_start_date_arr = (datetime_arr - pd.Timestamp(ANCHOR_DATE)).days

    return (
        solar_hour_angle_arr,
        solar_day_angle_arr,
        days_from_start_date_arr,
    )

def calculate_clear_sky_output(
        datetime_arr, latitude, longitude, time_difference_from_UTC
):
    """

    Args:
        datetime_arr(pd.DatetimeIndex)
        latitude(float): Latitude to be used to calculate solar elevation
        longitude(float): Longitude to be used to calculate local solar time in degrees. East->postive, West->Negative
        time_difference_from_from_UTC(int/float): Time-difference (in hours) between local time and
            Universal Coordinated TIme (UTC)

    Returns:
        clear_sky_output_df: Direct normal irradiation (W/m^2) time series at given latitude and longitude

    Reference for formulae:C.B.Honsberg and S.G.Bowden, “Photovoltaics Education Website,” www.pveducation.org, 2019
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
    solar_hour_angle_in_radians_arr = np.deg2rad(solar_hour_angle_arr)  # radians

    # Calculate solar declination
    solar_declination_arr = -23.5 * np.cos(np.deg2rad((360 / 365) * (day_of_year_arr + 10)))  # degrees
    solar_declination_in_radians_arr = np.deg2rad(solar_declination_arr)  # radians

    # Calculate solar altitude in each period
    latitude_in_radians = np.deg2rad(latitude)  # radians
    solar_elevation_angle_in_radians_arr = np.arcsin(
        np.sin(solar_declination_in_radians_arr) * np.sin(latitude_in_radians) + \
        np.cos(solar_declination_in_radians_arr) * np.cos(latitude_in_radians) * np.cos(solar_hour_angle_in_radians_arr)
    ) # radians

    # Calculate normalized clear sky output (proportional to sin(solar elevation angle))
    clear_sky_output_arr = np.sin(solar_elevation_angle_in_radians_arr)  # W/m^2
    # clear_sky_output cannot be negative; correct negative values to 0
    zeros = np.zeros(len(clear_sky_output_arr)) + 0.001 # Small constant added to avoid divide by zero errors
    clear_sky_output_arr = np.max([clear_sky_output_arr, zeros], axis=0)

    # Create clear_sky_output dataframe
    clear_sky_output_df = pd.DataFrame(index=datetime_arr, columns=["clear_sky_output"], data=np.array(clear_sky_output_arr))

    return clear_sky_output_df

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
    lag_terms_timeshift = pd.Series(np.arange(max_num_lag_terms, 0)) * pd.Timedelta(
        raw_data_freq
    )
    lead_terms_timeshift = pd.Series(
        np.arange(1, max_num_lead_terms + 1)
    ) * pd.Timedelta(raw_data_freq)
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


def infer_time_step(df):
    """

    Args:
        df: pd.DataFrame. A dataframe with an index that presumably have an innate frequency. Augment the pd.infer_freq
         function in the pandas package and make the return of type pd.Timedelta

    Returns: time_step: pd.Timedelta. The period of time between two rows of the input df.

    """
    # Determine time-step size in the data for this feature
    freq = pd.infer_freq(df.index)
    # If inferred frequency is 1 of something, say 1 min or 1 H, it will just be represented
    # as "T" or "H". Need to turn it into "1T" or "1H" to interpret as a number using the Timedelta function
    if freq[0].isdigit():
        time_step = pd.Timedelta(freq)
    else:
        time_step = pd.Timedelta("1" + freq)

    return time_step

def create_persistence_forecast(df_data_settings,
                                feature_type,
                                response_lead_term=response_lead_term,
                                ML_time_step=ML_time_step,
                                latitude=latitude,
                                longitude=longitude,
                                time_difference_from_UTC=time_difference_from_UTC,
                                ):
    # START persistence forecasting function

    # Get actuals filename from which to generate persistence forecasts
    actuals_filename = df_data_settings.loc[
        (df_data_settings["Feature"] == feature_type) & (df_data_settings["Type"] == "Actual")
    ].index[0] + ".csv"

    # Read in actuals data
    df_actuals = pd.read_csv(
        os.path.join(dir_str.data_checker_dir, actuals_filename),
        index_col=COL_NAME_FOR_DT,
        parse_dates=True,
        infer_datetime_format=True,
    )

    # Determine time step interval for this feature
    actuals_time_step = infer_time_step(df_actuals)

    # Correct actuals time step to match ML time step
    if actuals_time_step != ML_time_step:

        if actuals_time_step < ML_time_step: # Actuals data has higher frequency; downsample
            df_actuals = df_actuals.resample(ML_time_step).asfreq()

        elif actuals_time_step > ML_time_step: # Actuals data has lower frequency; upsample
            df_actuals = df_actuals.resample(ML_time_step).pad()

    if feature_type == "Wind":
        # Generate persistence forecast for wind

        # Shift actuals forward by # intervals by which response variable (forecast error) leads current time
        df_forecast = df_actuals.shift(periods=response_lead_term) # At some point, have user specify forecast horizon
        df_forecast.iloc[0:response_lead_term+1] = df_forecast.iloc[0:response_lead_term+1].bfill()
        # Rationale is that forecast for T + response_lead_term is value of actuals at current time
        # I.e. forecast for current time is value of actuals at T - response_lead_term

    elif feature_type == "Solar":
        # Generate persistence forecast for solar

        # Shift actuals forward by # intervals by which response variable (forecast error) leads current time
        df_forecast = df_actuals.shift(periods=response_lead_term)
        df_forecast.iloc[0:response_lead_term + 1] = df_forecast.iloc[0:response_lead_term + 1].bfill()
        # Rationale is that forecast for T + response_lead_term is value of actuals at current time
        # I.e. forecast for current time is value of actuals at T - response_lead_term

        # Adjust forecasts based on expected change in clear sky output
        clear_sky_output_df = calculate_clear_sky_output(
            df_actuals.index, latitude, longitude, time_difference_from_UTC
        )

        # Calculate adjustment factors
        adjustment_df = clear_sky_output_df/clear_sky_output_df.shift(periods=response_lead_term)
        adjustment_df.iloc[0:response_lead_term + 1] = adjustment_df.iloc[0:response_lead_term + 1].bfill()
        # Cap adjustment factors at 2
        adjustment_df.clip(lower=0.0, upper=2.0, inplace=True)
        # Apply adjustment factors to forecasts
        df_forecast[COL_NAME_FOR_VALUE] *= adjustment_df.values.flatten()

    # Save forecast
    persistence_forecast_filename = "{}_persistence_forecast_T+{:.0f}.csv".format(
        feature_type, response_lead_term
    )
    df_forecast.to_csv(os.path.join(dir_str.data_checker_dir, persistence_forecast_filename))

    # Update data settings dataframe index
    df_data_settings["new_index"] = df_data_settings.index
    df_data_settings.loc[
        (df_data_settings["Feature"] == feature_type) & (df_data_settings["Type"] == "Forecast"), "new_index"
    ] = persistence_forecast_filename.strip(".csv")
    df_data_settings.set_index(["new_index"], inplace=True)
    df_data_settings.index.rename("Data Source", inplace=True)

    return df_data_settings


def main(
    model_name=model_name,
    ML_time_step=ML_time_step,
    response_col_names=response_col_names,
    lag_term_start_predictors=lag_term_start_predictors,
    lag_term_end_predictors=lag_term_end_predictors,
    lag_term_step_predictors=lag_term_step_predictors,
    response_lead_term=response_lead_term,
    longitude=longitude,
    latitude=latitude,
    time_difference_from_UTC=time_difference_from_UTC,
    df_data_settings=df_data_settings,
):
    # ==== 0. Read in each time-series feature, output from the data-checker and ensure it matches ML time-step ====
    # Paths to read raw data files from and to store outputs in. Defined in the dir_structure class in utility
    dir_str = utility.Dir_Structure(model_name=model_name)

    # Check whether to create persistence forecasts for certain features
    if "persistence" in df_data_settings.index:
        # Generate persistence forecasts, save as CSV, and correct df_data_settings with new feature name
        temp = df_data_settings.set_index(["Feature"], append=True).loc["persistence"]
        # Get feature types (e.g. load, wind, solar) for which to generate persistence forecasts
        for feature_type in temp.index:
            df_data_settings = create_persistence_forecast(df_data_settings, feature_type)

    # Initialize df to hold collected data for ML model
    raw_data_df = pd.DataFrame()

    # Iterate over each feature output from data-checker
    for feature_name in df_data_settings.index:

        # prompt user about progress
        print("Reading in feature: {}".format(feature_name))

        feature_df = pd.read_csv(
            os.path.join(dir_str.data_checker_dir, feature_name + ".csv"),
            index_col=COL_NAME_FOR_DT,
            parse_dates=True,
            infer_datetime_format=True,
        )

        # Replace invalid rows with None and only keep the value column
        feature_df_validity = feature_df[COL_NAME_FOR_VALIDITY_FLAG]
        feature_df.loc[~feature_df_validity, COL_NAME_FOR_VALUE] = None
        feature_df = feature_df[COL_NAME_FOR_VALUE].rename(feature_name)

        # Determine time step interval for this feature
        feature_time_step = infer_time_step(feature_df)

        # ==== Option 1 of 3 ====
        # If the feature time-step matches that needed for ML inputs, do nothing
        # ==== Option 2 of 3 ====
        # If the time step is shorter/more frequent than that desired, create multiple features for each ML time step
        # Say we have 5-min features and ML inputs are on a 15-min resolution. Then, create three 15-min feature stream
        # from this 5-min feature stream corresponding to the three 5 minute intervals in the 15 minute.
        if feature_time_step < ML_time_step:
            assert (ML_time_step % feature_time_step).total_seconds() == 0, (
                "When ML model's temporal time step is longer than that of the raw feature data, it must be a multiple "
                "of the feature timestep. Not the case for {}".format(feature_name)
            )
            feature_df = feature_df.resample(ML_time_step).asfreq() # Downsample
        # ==== Option 3 of 3 ====
        # If feature time-step is longer than desired, replicate feature
        # For eg, if ML time step is 15 min and feature df has values on hourly resolution, we will repeat the feature
        # value 4 times, once for each 15 min interval in the given hour
        elif feature_time_step > ML_time_step:
            assert (feature_time_step % ML_time_step).total_seconds() == 0, (
                "When ML model's temporal time step is shorter than that of the raw feature data, it must be a factor "
                "of the feature time step. Not the case for {}".format(feature_name)
            )
            feature_df = feature_df.resample(ML_time_step).pad() # Upsample and pad

        # Merge the current feature with the rest of the features
        raw_data_df = pd.concat((raw_data_df, feature_df), axis=1, join="outer")

    # ==== 1. Pad raw date for downstream manipulation ====
    # Pad the raw data with NaNs in both the lag and lead direction for downstream data manipulation
    raw_data_df, raw_data_start_idx, raw_data_end_idx = pad_raw_data_w_lag_lead(
        raw_data_df,
        lag_term_start_predictors,
        lag_term_end_predictors,
        response_lead_term,
    )

    # ==== 2. Add in calendar terms for the raw data ====
    print("Calculating calendar-based predictors....")
    (
        raw_data_df[COL_NAME_HOUR_ANGLE],
        raw_data_df[COL_NAME_DAY_ANGLE],
        raw_data_df[COL_NAME_DAYS_IDX],
    ) = calculate_calendar_based_predictors(
        raw_data_df.index,
        longitude,
        time_difference_from_UTC,
    )

    ### CAISO specific
    # Calculate ML interval ids w.r.t each rtpd interval. In this case we often set ML interval = RTD interval
    # For eg, rtd interval starting 12:00 = 0, 12:05 = 1, 12:10 = 2, if rtpd interval spans from 12:00 to 12:15
    # this feature is meaningless when the ML time step is just RTPD interval
    COL_NAME_INTERVAL_ID = "5_Min_Interval_ID"

    if ML_time_step != pd.Timedelta("15T"):
        rtpd_to_ML_multitude = int(pd.Timedelta("15T") / ML_time_step)
        raw_data_df[COL_NAME_INTERVAL_ID] = (
            (raw_data_df.index - pd.Timestamp(ANCHOR_DATE)) // ML_time_step
        ) % rtpd_to_ML_multitude

    # ==== 3. Add in net-load forecast difference and load, solar, wind (if multi-obj) forecast difference
    # for the raw data. These will be used as response variable(s) ====
    print("Calculating response(s)....")
    response_df = calculate_response_variables(raw_data_df, response_col_names, df_data_settings)
    raw_data_df = pd.concat([raw_data_df, response_df], axis=1)

    # Revise the lag term array since we are extending the original data
    for feature in raw_data_df.columns:
        if feature not in lag_term_step_predictors.index:
            # All other features get same lag term interval start/end/step
            lag_term_start_predictors.loc[feature] = response_lead_term
            lag_term_end_predictors.loc[feature] = response_lead_term
            lag_term_step_predictors.loc[feature] = response_lead_term

    # Revise the lag term array since we are extending the original data
    #num_feature_ext = len(raw_data_df.columns) - len(lag_term_start_predictors)
    # for the response term, since there is just one for each category, the start, end, and step all equal to lead
    #response_lead_term_all = np.ones(num_feature_ext, dtype=int) * response_lead_term
    #lag_term_start_all = np.hstack((lag_term_start_predictors, response_lead_term_all))
    #lag_term_end_all = np.hstack((lag_term_end_predictors, response_lead_term_all))
    #lag_term_step_all = np.hstack((lag_term_step_predictors, response_lead_term_all))

    # ==== 4. Using vectorized operations to construct lag terms ====
    print("Creating trainval samples for all time-points ....")
    # Initialize collectors to hold (and later save) trainval data in
    trainval_data_df = pd.DataFrame(
        None, index=raw_data_df.index[raw_data_start_idx:raw_data_end_idx]
    )

    # Collect lag term predictors for all trainval samples
    # Iterate over each type of lag term predictor
    for feature in raw_data_df.columns:

        # obtain lag term start and end offset, as well as step size for a certain feature
        lag_term_start = lag_term_start_predictors.loc[feature]
        lag_term_end = lag_term_end_predictors.loc[feature]
        lag_term_step = lag_term_step_predictors.loc[feature]

        # Iterate over each time step for current predictor type
        for time_step in range(lag_term_start, lag_term_end + 1, lag_term_step):
            label = "{}_T{:+}".format(feature, time_step)
            trainval_data_df[label] = (
                raw_data_df[feature]
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
    # Only retain trainval samples wherein predictor(s) and response(s) are both valid
    trainval_data_df = trainval_data_df.dropna()

    # Separate predictors (model inputs) from response (model output(s))
    print("Saving files......")
    response_col_labels = [
        "{}_T{:+}".format(name, response_lead_term) for name in response_col_names.values()
    ]
    trainval_outputs_df = trainval_data_df.loc[:, response_col_labels].copy()
    trainval_data_df = trainval_data_df.drop(columns=trainval_outputs_df.columns)

    # Save trainval samples
    trainval_data_df.to_pickle(dir_str.input_trainval_path)
    trainval_outputs_df.to_pickle(dir_str.output_trainval_path)

    print("All done!")


# run as a script
if __name__ == "__main__":
    main()
