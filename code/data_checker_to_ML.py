import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## User inputs and constants

# Paths and filenames
# TODO: Consider integrating these into utility.py after Yuchi signs off on this script being part of ML package
base_path = os.path.dirname(os.getcwd())
path_to_data_checker_outputs = os.path.join(base_path, r"data_checker_outputs")
path_to_ML_inputs = os.path.join(base_path, r"data\raw_data")
filename_for_input_values = "input_values_for_M_by_N_creating_script.csv"
filename_for_input_validity_flags = "input_validity_flags_for_M_by_N_creating_script.csv"

# Declare files that data-checker produces but aren't to be used to create inputs for ML
# TODO: Should the data checker script be updated to put files to be used in ML model in a sub-folder?
files_to_ignore = ["summary_all_files.csv", "archive"]

# Other constants
# TODO: The challenge with implying this from the data-checker outputs is that different output files may span across different time periods. Which one to rely on?
# Temporal characteristics required for making trainval set
start_date = "01-01-2017" # Inclusive
end_date = "01-01-2020" # Exclusive
ML_time_step = "5T" # T implies minutes

# Column names corresponding to those in data-checker output files
col_name_for_value = "Forecast_Interval_Avg_MW"
col_name_for_validity_flag = "valid_all_checks"
datetime_col_name = "Datetime_Interval_Start"

# Data to be used in correlation analysis
# Directions associated whether we wish to retain any features by default irrespective of correlation results
# TODO: Consider placing these in file potentially holding lag-lead term #s going forward
mandatory_feature = [True, True, True, True, True, True]
# How many additional features to add in based on correlation analysis?
max_num_optional_features = 0

## Script/Funcs outside user control

# Helper function(s)
def merge_df(df_numeric_1, df_numeric_2, df_bool_1, df_bool_2):
    """
    Perform outer merging of dataframes
    :param df_numeric_1:DataFrame holding numeric values
    :param df_numeric_2:DataFrame holding numeric values, to be merged with df_1
    :param df_bool_1:DataFrame holding booleans that highlight validity of df_numeric_1
    :param df_bool_2:DataFrame holding booleans that highlight validity of df_numeric_2, to be merged with df_1
    :return: df_numeric_merged, df_bool_merged: DataFrames resulting from outer merge operation
    """
    df_numeric_merged = df_numeric_1.merge(df_numeric_2, right_index=True, left_index=True, how="outer")
    df_bool_merged = df_bool_1.merge(df_bool_2, right_index=True, left_index=True, how="outer")
    # In case index of the two dfs didn't match (can happen if some feature has a longer/shorter historical
    # record), fill the resulting nans with 0 and FALSE resp. Numeric data will be used in conjunction with bool data
    # later. FALSE data will be effectively ignored. So, we don't need more sophisticated replacements for nans
    df_numeric_merged = df_numeric_merged.fillna(0.0)
    df_bool_merged = df_bool_merged.fillna("FALSE")
    return df_numeric_merged, df_bool_merged

def identify_highly_correlated_features(df, optional_feature, max_num_optional_features):
    """
    Calculates and plots pair-wise correlation matrix for each column in df. Then identifies most highly
    correlated optional features (each represented by a column in the df) to remove
    :param df: DataFrame - each column holds values corresponding to one feature
    :param optional_feature: Array holding booleans, length and order corresponds to column names in df. True if
    a particular column is optional and can be dropped if found to be highly correlated with the others
    :param max_num_optional_features: Int-How many of the optional features can be retained?
    :return: features_to_remove: List of features to remove from ML inputs' data
    """
    # Calculate and plot correlation matrix
    corr_matrix = df.corr()
    plt.imshow(corr_matrix, cmap="Reds")
    plt.colorbar()
    plt.yticks(np.arange(len(corr_matrix.index)), corr_matrix.index)
    plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation='vertical')
    plt.title("Correlation Matrix For All Features", fontweight="bold")
    plt.show()
    # Of the optional features, identify the ones most correlated with the others
    optional_features_corr_sum = corr_matrix.sum(axis=1).loc[optional_feature].sort_values()
    features_to_remove = optional_features_corr_sum.index[max_num_optional_features:]
    return features_to_remove

# Initialize df to hold collated data for ML model
ML_inputs_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq=ML_time_step, closed="left"))
ML_inputs_validity_df = ML_inputs_df.copy()

# This will be used to keep account of which features were turned into multiple ones due to misalignment in time-step
final_feature_count = np.ones_like(mandatory_feature, dtype=int)
feature_idx = 0
# Iterate over each feature output from data-checker
for file in os.listdir(path_to_data_checker_outputs):
    feature_name = file.strip(".csv")
    print("On {}....".format(feature_name))
    if file in files_to_ignore:
        print("Skipping {}. Not assumed to be holding inputs for ML model".format(file))
        continue
    feature_df = pd.read_csv(os.path.join(path_to_data_checker_outputs, file),
                             usecols=[datetime_col_name, col_name_for_value, col_name_for_validity_flag],
                             index_col=datetime_col_name,
                             parse_dates=True)
    # Determine time-step size in the data for this feature
    # Timedelta object doesn't have a minutes attribute, so we'll make do with the seconds attribute
    feature_time_step = (feature_df.index[1] - feature_df.index[0]).seconds
    # Determine time-step required for ML model
    ML_inputs_time_step = (ML_inputs_df.index[1] - ML_inputs_df.index[0]).seconds

    # ==== Option 1 of 3 ====
    # If the feature time-step size matches that needed for ML inputs, place the feature into the ML inputs df
    # without any changes
    if feature_time_step == ML_inputs_time_step:
        ML_inputs_df, ML_inputs_validity_df = merge_df(ML_inputs_df, feature_df[col_name_for_value],
                                                       ML_inputs_validity_df, feature_df[col_name_for_validity_flag])
        # Rename from generic col name to feature-specific name
        ML_inputs_df = ML_inputs_df.rename(columns={col_name_for_value:feature_name})
        ML_inputs_validity_df = ML_inputs_validity_df.rename(columns={col_name_for_validity_flag: feature_name})
    # ==== Option 2 of 3 ====
    # If the time step is shorter than that desired, create multiple features
    # Lets say we have 5-min features and ML inputs are on a 15-min resolution. Then, create three 15-minutely features
    # out of the one 5-min feature currently being assessed
    elif feature_time_step < ML_inputs_time_step:
        assert ML_inputs_time_step % feature_time_step == 0, "ML time-step needs to be equal to or a perfect multiple of " \
                                                             "feature time-step. Not the case for {}".format(feature_name)
        num_substeps = int(ML_inputs_time_step / feature_time_step)
        # Replicate time-stamps in feature_df such that all sub-steps within a ML step have same datetime
        replicated_time_stamps = feature_df.index[::num_substeps].repeat(num_substeps)
        last_idx = len(feature_df.index)
        # Any stray datetimes that don't fit within the final ML step will be ignored
        feature_df.index = replicated_time_stamps[:last_idx]
        for start_idx in range(num_substeps):
            ML_inputs_df, ML_inputs_validity_df = merge_df(ML_inputs_df,
                                                           feature_df[col_name_for_value].iloc[start_idx::num_substeps],
                                                           ML_inputs_validity_df,
                                                           feature_df[col_name_for_validity_flag].iloc[start_idx::num_substeps])
            # Rename from generic col name to feature-specific name
            ML_inputs_df = ML_inputs_df.rename(columns={col_name_for_value:feature_name+"_"+str(start_idx)})
            ML_inputs_validity_df = ML_inputs_validity_df.rename(columns={col_name_for_validity_flag:feature_name+"_"+str(start_idx)})
        # Make note of the fact that we've created additional features
        final_feature_count[feature_idx] = num_substeps
    # ==== Option 3 of 3 ====
    # If feature time-step is longer than desired, replicate feature
    # For eg, if ML time step is 15 min and feature df has values on hourly resolution, we will repeat the feature value
    # 4 times, once for each 15 min interval in the given hour
    else:
        assert feature_time_step % ML_inputs_time_step == 0, "Feature time-step needs to be equal to or a perfect " \
                                                             "multiple of the ML time-step. Not the case for " \
                                                             "{}".format(feature_name)
        num_substeps = int(feature_time_step / ML_inputs_time_step)
        repeated_feature_df = pd.DataFrame(np.repeat(feature_df.values, num_substeps, axis=0))
        repeated_feature_df.columns = feature_df.columns
        # Add substeps to feature_df datetime index
        new_datetime_index = pd.date_range(start=feature_df.index[0], periods=len(feature_df.index) * num_substeps,
                                           freq=str(ML_inputs_time_step) + "S")
        repeated_feature_df.index = new_datetime_index
        ML_inputs_df, ML_inputs_validity_df = merge_df(ML_inputs_df, repeated_feature_df[col_name_for_value],
                                                       ML_inputs_validity_df, repeated_feature_df[col_name_for_validity_flag])
        # Rename from generic col name to feature-specific name
        ML_inputs_df = ML_inputs_df.rename(columns={col_name_for_value: feature_name})
        ML_inputs_validity_df = ML_inputs_validity_df.rename(columns={col_name_for_validity_flag: feature_name})
    feature_idx += 1

# Ensure newly created features get tagged as optional/mandatory as appropriate
mandatory_feature = np.repeat(mandatory_feature, final_feature_count)
optional_feature = np.logical_not(mandatory_feature)
# Conduct correlation analysis and identify features to be removed, if any
print("Performing correlation analysis to remove optional features highly correlated with the others....")
features_to_remove = identify_highly_correlated_features(ML_inputs_df, optional_feature, max_num_optional_features)
print("Features to be removed- {}....".format(features_to_remove.values))
# Drop these feature from ML inputs' df
ML_inputs_df = ML_inputs_df.drop(columns=features_to_remove)
ML_inputs_validity_df = ML_inputs_validity_df.drop(columns=features_to_remove)

# Save ML inputs
print("Saving all ML inputs....")
ML_inputs_df.to_csv(os.path.join(path_to_ML_inputs, filename_for_input_values))
ML_inputs_validity_df.to_csv(os.path.join(path_to_ML_inputs, filename_for_input_validity_flags))
print("All done")















