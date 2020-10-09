#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#  User inputs
# 

# In[8]:


# Paths to raw data files and store outputs
path_to_raw_data = os.path.join(os.getcwd(), "dummy_files_to_test_code_with")
raw_data_file_name = "raw_data_dummy.csv"
path_to_store_outputs_at = os.path.join(os.getcwd(), "dummy_files_to_test_code_with") 
trainval_inputs_data_file_name = "trainval_inputs.npy"
trainval_output_data_file_name = "trainval_output.npy"
datetimes_for_trainval_data_file_name = "trainval_datetimes.npy"

# Constants
num_of_lag_terms = 2 # Excluding T0. If = 2, implies T-1, T-2
# TODO: Can this be determined based on num_lag_terms?
num_of_predictors = 20
# # Not currently being used. Revisit if need be
# forecast_error_at_T0_available = False # If False, we will not use forecast error at T0. Only at T0-1, T0-2, ...
# Those associated with calculating calendar terms - currently solar hour angle and day angle
longitude = -119.4179 # Roughly passing through the center of CA
time_difference_from_UTC = -8 # hours. Timestamps for input data are in PST
# TODO: Consider having option to disallow overlaps between subsequent training samples

# Stuff in raw data file
# Main types of timerseries data- Names should match those in raw data file
main_timeseries_data = ["Load", "Solar", "Wind"]
# Suffixes for column names used in raw data file
forecast_suffix = "_Forecast"
actual_suffix = "_Actual"
ignore_flag_suffix = "_Ignore_Flag"
# Other column names
datetime_col_name = "Datetime"


# Constants for use in script that don't need to be user defined

# In[9]:


forecast_error_suffix = "_Forecast_Error"
hour_angle_col_name = "Hour_Angle"
day_angle_col_name = "Day_Angle"


# Reading in raw data, doing any modifications/edits before creating trainval samples 

# In[11]:


# Read in raw data
raw_data_df = pd.read_csv(os.path.join(path_to_raw_data, raw_data_file_name))
num_of_time_points = raw_data_df.shape[0]

# Add other columns to raw data - forecast error, calendar terms, etc
for mtd in main_timeseries_data:
    raw_data_df[mtd + forecast_error_suffix] = raw_data_df[mtd + actual_suffix] - raw_data_df[mtd + forecast_suffix]
    raw_data_df[mtd + forecast_error_suffix + ignore_flag_suffix] = raw_data_df[mtd + actual_suffix + ignore_flag_suffix] &                                                                 raw_data_df[mtd + forecast_suffix + ignore_flag_suffix]
                                                                
# Calculation of calendar terms
print("Calculating Solar Hour & Day Angles....")
raw_data_df[hour_angle_col_name], raw_data_df[day_angle_col_name] = calculate_solar_hour_and_day_angles(pd.to_datetime(raw_data_df[datetime_col_name].values),                                                                longitude, time_difference_from_UTC)
print("Done")

# Initialize collectors to hold (and later save) trainval data in
trainval_inputs_data_arr = np.empty((num_of_predictors, num_of_time_points), dtype = float)
trainval_inputs_data_arr[:] = np.nan
trainval_output_data_arr = np.empty((1, num_of_time_points), dtype = float)
trainval_output_data_arr[:] = np.nan
# Datetimes are just stored for reference. Not part of the trainval inputs/outputs - atleast not at the moment
datetimes_for_trainval_data_arr = np.empty((num_of_lag_terms + 2, num_of_time_points), dtype = object) # +2 because some values at T0 are predictors too and response is at T0 + 1 
datetimes_for_trainval_data_arr[:] = np.nan


# Creating and saving trainval samples 

# In[12]:


# Iterate over each potential time point in raw data that we can create a trainval sample for
for time_pt in range(num_of_lag_terms, num_of_time_points - 1):
    time_pt_can_be_converted_to_trainval_sample = True
    # Check validity of data at all time points tied to this one in a given trainval sample
    # First check validity of forecasts
    for tied_time_pt in range(time_pt - num_of_lag_terms, time_pt + 2):
        for mtd in main_timeseries_data:
            if raw_data_df.loc[tied_time_pt, mtd + forecast_suffix + ignore_flag_suffix]:
                time_pt_can_be_converted_to_trainval_sample = False
                print("{} Forecast invalid at {}".format(mtd, raw_data_df.loc[tied_time_pt, datetime_col_name]))
                break
#     # Next check validity of actuals
#     if time_pt_can_be_converted_to_trainval_sample:
#         for mtd in main_timeseries_data:
#             if raw_data_df.loc[time_pt + 1, mtd + actual_suffix + ignore_flag_suffix]:
#                 time_pt_can_be_converted_to_trainval_sample = False
#                 break
    # Next check validity of forecast errors
    if time_pt_can_be_converted_to_trainval_sample:
            for tied_time_pt in range(time_pt - num_of_lag_terms, time_pt):
                for mtd in main_timeseries_data:
                    if raw_data_df.loc[tied_time_pt, mtd + forecast_error_suffix + ignore_flag_suffix]:
                        time_pt_can_be_converted_to_trainval_sample = False
                        print("{} Forecast error invalid at {}".format(mtd, raw_data_df.loc[tied_time_pt, datetime_col_name]))
                        break
    
    # If all data points needed are valid, turn current time pt into a trainval sample
    if time_pt_can_be_converted_to_trainval_sample:
        # First collect inputs for this trainval sample
        # Forecasts
        start_idx = 0
        num_forecast_terms_in_sample = num_of_lag_terms + 2
        for mtd in main_timeseries_data:
            trainval_inputs_data_arr[start_idx:start_idx + num_forecast_terms_in_sample, time_pt] = raw_data_df.loc[time_pt - num_of_lag_terms:time_pt + 1, mtd + forecast_suffix].values
            start_idx += num_forecast_terms_in_sample
        # Forecast errors
        num_forecast_error_terms_in_sample = num_of_lag_terms
        for mtd in main_timeseries_data:
            trainval_inputs_data_arr[start_idx:start_idx + num_forecast_error_terms_in_sample, time_pt] = raw_data_df.loc[time_pt - num_of_lag_terms:time_pt - 1, mtd + forecast_error_suffix].values
            start_idx += num_forecast_error_terms_in_sample
        # Calendar terms
        trainval_inputs_data_arr[start_idx, time_pt] = raw_data_df.loc[time_pt + 1, hour_angle_col_name]
        trainval_inputs_data_arr[start_idx + 1, time_pt] = raw_data_df.loc[time_pt + 1, day_angle_col_name]
        
        # Next collect output for this trainval sample
        # TODO: Once loss function is defined, determing what this will be. Alternatively,c an do this in ML model script
        trainval_output_data_arr[0, time_pt] = -99999
        
        # Finally, record all time pts present in this training sample for future reference/checks
        datetimes_for_trainval_data_arr[:, time_pt] = raw_data_df.loc[time_pt - num_of_lag_terms:time_pt + 1, datetime_col_name].values
        
    # Print progress periodically
    for prog_idx in range(1,5):
        if time_pt == int(0.25 * prog_idx * num_of_time_points):
            print("{}% there.......".format(25*prog_idx))
        
# Delete time-points that were invalid from the trainval array
trainval_inputs_data_arr = trainval_inputs_data_arr[:,~np.all(np.isnan(trainval_inputs_data_arr), axis=0)]
trainval_output_data_arr = trainval_output_data_arr[:,~np.all(np.isnan(trainval_output_data_arr), axis=0)]

print("Saving trainval samples......")
# Save trainval samples
np.save(os.path.join(path_to_store_outputs_at, trainval_inputs_data_file_name), trainval_inputs_data_arr)
np.save(os.path.join(path_to_store_outputs_at, trainval_output_data_file_name), trainval_output_data_arr)
np.save(os.path.join(path_to_store_outputs_at, datetimes_for_trainval_data_file_name), datetimes_for_trainval_data_arr)
# TO DO: Delete this after debugging
np.savetxt(os.path.join(path_to_store_outputs_at, "trainval_inputs.csv"), trainval_inputs_data_arr, delimiter = ",", fmt='%f')
np.savetxt(os.path.join(path_to_store_outputs_at, "trainval_output.csv"), trainval_output_data_arr, delimiter = ",", fmt='%f')
# np.savetxt(os.path.join(path_to_store_outputs_at, "datetimes_for_trainval.csv"), datetimes_for_trainval_data_arr, delimiter = ",", fmt='%f')
print("Done!")


# Helper functions

# In[10]:


def calculate_solar_hour_and_day_angles(datetime_arr, longitude, time_difference_from_UTC):
    """
    Calculated solar hour and day angles for each time-point in the input time series
    
    Inputs: 
    datetime_arr(pd.DatetimeIndex)
    longitude(float): Longitude to be used to calculate local solar time in degrees. East->postive, West->Negative
    time_difference_from_from_UTC(int/float): Time-difference (in hours) between local time and 
    Universal Coordinated TIme (UTC)
    
    Output:
    solar_hour_angle_arr (Array of floats): Hour angle in degrees for each timepoint in datetime_arr
    solar_day_angle_arr (Array of floats): Day angle in degrees for each timepoint in datetime_arr
    
    Reference for formulae:https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time
    """
    # Steps leading up to calculation of local solar time
    day_of_year_arr = datetime_arr.dayofyear
    # Equation of time (EoT) corrects for eccentricity of earth's orbit and axial tilt
    solar_day_angle_arr = (360/365) * (day_of_year_arr - 81) # degrees
    solar_day_angle_in_radians_arr = np.deg2rad(solar_day_angle_arr) # radians
    EoT_arr = 9.87 * np.sin(2 * solar_day_angle_in_radians_arr) - 7.53 * np.cos(solar_day_angle_in_radians_arr) -               1.5 * np.sin(solar_day_angle_in_radians_arr) # minutes
    # Time correction sums up time difference due to EoT and longitudinal difference between local time
    # zone and local longitude
    local_std_time_meridian = 15 * time_difference_from_UTC # degrees
    time_correction_arr = 4 * (longitude - local_std_time_meridian) + EoT_arr # minutes
    # Calculate local solar time using local time and time correction calculated above
    local_solar_time_arr = datetime_arr.hour +                                     (datetime_arr.minute / 60) +                                     (time_correction_arr / 60) # hours
    # Calculate solar hour angle corresponding to the local solar time
    solar_hour_angle_arr = 15 * (local_solar_time_arr - 12) # degrees
        
    return solar_hour_angle_arr, solar_day_angle_arr


# Testing/debugging. DELETE after you are done

# In[14]:


raw_data_df


# In[ ]:




