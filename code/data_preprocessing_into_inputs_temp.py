# Temporary script used to create inputs for experiment where predictions need to happen on 5-min intervals but
# most recent 5-min data cannot be used.
import os
import pandas as pd
import numpy as np
import datetime

## User inputs

# Paths and file names
# Using previously created trainval inputs and outputs as starting points
base_path = os.path.dirname(os.getcwd())
path_to_trainval_inputs = os.path.join(base_path, r"data\rescue_v1_1_multi_objective\input_trainval.pkl")
path_to_trainval_outputs = os.path.join(base_path, r"data\raw_data\input_values_for_M_by_N_creating_script.csv")
path_to_15_to_5_min_feature_mapping = os.path.join(base_path, r"data\raw_data\15_to_5_min_feature_name_mapping.csv")
path_to_model_data = os.path.join(base_path, r"data")
model_name = "rescue_5_min_v2"

# Constants
rtpd_interval = 15 # minutes
rtd_interval = 5 # minutes
num_reps = int(rtpd_interval/rtd_interval)
# RTD ID to distinguish each RTD interval comprised within a single RTPD interval from each other
rtd_ids = [0, 1, 2]
# If False, only Net Load Forecast Error will be calculated and saved
multi_obj_flag = True

## Script to create trainval inputs and outputs for this particular experiment

# First post-process 15-min trainval inputs to get 5-min trainval inputs via repetition
trainval_inputs_15_min_df = pd.read_pickle(path_to_trainval_inputs)
trainval_inputs_df = pd.DataFrame(np.repeat(trainval_inputs_15_min_df.values, num_reps, axis=0))
trainval_inputs_df.columns = trainval_inputs_15_min_df.columns
trainval_inputs_df.index = np.repeat(trainval_inputs_15_min_df.index.values, num_reps)
# Add new column to distinguish the 3 RTD intervals within a single RTPD interval from each other
trainval_inputs_df["5_Min_Interval_ID_T+3"] = np.tile(rtd_ids, trainval_inputs_15_min_df.shape[0])
# Update datetimes to go from 15 to 5 min resolution
minutes_to_add = trainval_inputs_df["5_Min_Interval_ID_T+3"].values * rtd_interval
trainval_inputs_df.index = trainval_inputs_df.index + pd.Series([datetime.timedelta(minutes=int(x)) for x in minutes_to_add])
# Update column names so they're consistent with the other 5-min prediction experiment
column_name_dict = pd.read_csv(path_to_15_to_5_min_feature_mapping)
column_name_dict = dict(zip(column_name_dict["15_min_name"].values, column_name_dict["5_min_name"].values))
trainval_inputs_df = trainval_inputs_df.rename(columns=column_name_dict)

# Next get trainval outputs for this experiment
trainval_outputs_df = pd.read_csv(path_to_trainval_outputs, index_col=0, parse_dates=True)
# Remove RTPD interval minutes from datetime index to be consistent with trainval data previously created
trainval_outputs_df.index = trainval_outputs_df.index - pd.Timedelta(minutes=rtpd_interval)
# Calculate forecast error
trainval_outputs_df["Load_Forecast_Error_T+3"] = trainval_outputs_df["Load_RTPD_Forecast"] - trainval_outputs_df["Load_RTD_Forecast"]
trainval_outputs_df["Solar_Forecast_Error_T+3"] = trainval_outputs_df["Solar_RTPD_Forecast"] - trainval_outputs_df["Solar_RTD_Forecast"]
trainval_outputs_df["Wind_Forecast_Error_T+3"] = trainval_outputs_df["Wind_RTPD_Forecast"] - trainval_outputs_df["Wind_RTD_Forecast"]
trainval_outputs_df["Net_Load_Forecast_Error_T+3"] = trainval_outputs_df["Load_Forecast_Error_T+3"] - trainval_outputs_df["Solar_Forecast_Error_T+3"] - trainval_outputs_df["Wind_Forecast_Error_T+3"]
if multi_obj_flag:
    trainval_outputs_df = trainval_outputs_df.loc[trainval_inputs_df.index, ["Load_Forecast_Error_T+3", "Solar_Forecast_Error_T+3", "Wind_Forecast_Error_T+3", "Net_Load_Forecast_Error_T+3"]]
else:
    trainval_outputs_df = trainval_outputs_df.loc[trainval_inputs_df.index, "Net_Load_Forecast_Error_T+3"]

# Save created trainval inputs and outputs
if not os.path.isdir(os.path.join(path_to_model_data, model_name)):
    os.makedirs(os.path.join(path_to_model_data, model_name))
trainval_inputs_df.to_pickle(os.path.join(path_to_model_data, model_name, "input_trainval.pkl"))
trainval_outputs_df.to_pickle(os.path.join(path_to_model_data, model_name, "output_trainval.pkl"))





