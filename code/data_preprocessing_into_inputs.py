import os
import numpy as np
import pandas as pd
import pathlib
from utility import Dir_Structure
from parse_excel_configs import ExcelConfigs
from calendrical_predictors import CalendricalPredictors

# ==== Constants  ====
# Column names used in data-checker output files
COL_NAME_VALUE = "Interval_Avg_Quantity"
COL_NAME_VALIDITY = "valid_all_checks"
COL_NAME_DATETIME = "Datetime_Interval_Start"

# Input excel name
INPUT_EXCEL_NAME = pathlib.Path("RSERVE_Input_v1.xlsx")

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

    return response_values_df


def read_all_timeseries(configs, dir_str):
    """

    Returns:

    """
    ts_attrs = configs.timeseries_attributes  # alias

    # Reading in time series and matching them to the frequency needed
    ts_data_df = pd.DataFrame()  # empty container for all features and sub-features
    # Iterate over each feature output from data-checker
    for feature_name in ts_attrs.index:

        print("Reading in " + feature_name + "...")
        feature_df = pd.read_csv(
            os.path.join(dir_str.data_checker_dir, ts_attrs.loc[feature_name, "File Name"]),
            index_col=COL_NAME_DATETIME,
            parse_dates=True,
        )
        # match the feature frequency with the required frequency of ML problem
        ts_data_one = match_frequency(feature_df, feature_name, configs.sample_interval)

        if ts_data_one.shape[1] >= 2:  # when sub-features has been made
            # The match frequency process sometimes change the name and amount of time series
            for i, term_configs in enumerate([configs.lag_term_configs, configs.lead_term_configs]):
                col_to_check = "Is Input?" if i == 0 else "Is Output?"
                if ts_attrs.loc[feature_name, col_to_check]:
                    term_configs.loc[ts_data_one.columns] = term_configs[feature_name].values  # add the new
                    term_configs.drop(feature_name, axis=0, inplace=True)  # drop the old

        # collect each individual feature or sub-feature into the total timeseries data dataframe
        ts_data_df = pd.concat([ts_data_df, ts_data_one], axis=1, join="outer")

    return ts_data_df


def match_frequency(feature_df, feature_name, sample_interval):
    """
    Unstack or pad the original feature in order to achieve desired frequency.

    Args:
        feature_df: pd.DataFrame. the dataframe of the feature read in from hard drive, following data checker output
        format
        feature_name: str. Name of the feature
        sample_interval: int. The time step of the ML model expressed in seconds.

    Returns:
        ts_data_srs: pd.Series of pd.DataFrame. Time series data

    """

    # Embed info about validity, so we can use the ts_data_df alone going forward
    feature_df.loc[~feature_df[COL_NAME_VALIDITY], COL_NAME_VALUE] = None
    # Rename from generic col name to feature-specific name
    ts_data_one = feature_df[COL_NAME_VALUE].rename(feature_name)

    # Determine time-step size in the data for this feature
    feature_freq = pd.infer_freq(ts_data_one.index)
    assert feature_freq is not None, "Raw data does not have equally spaced index! Cannot discern Frequency!"

    # If inferred frequency is 1 of something, say 1 min or 1 H, it will just be represented
    # as "T" or "H". Need to turn it into "1T" or "1H" to interpret as a number using the Timedelta function
    if feature_freq[0].isdigit():
        feature_time_step = pd.Timedelta(feature_freq)
    else:
        feature_time_step = pd.Timedelta("1" + feature_freq)

    # TODO: The script takes care of frequency mismatch, but not starting time mismatch.

    # ==== Option 1 of 3 ====
    # If the feature time-step size matches, place the feature into the ML inputs df without any changes
    if feature_time_step == sample_interval:
        ts_data_one = ts_data_one.to_frame()

    # ==== Option 2 of 3 ====
    # If the input is more frequent than desired, create multiple features
    # E.g. Given 5-min features and ML inputs are on a 15-min resolution. Then, create three 15-minutely features
    # out of each one of the 5-min feature currently being assessed
    elif feature_time_step < sample_interval:
        # calculate number of sub steps and organize feature frequency into sth that can be obtained
        # through dividing ML time step with an integer
        num_sub_steps = sample_interval // feature_time_step
        ts_data_one = ts_data_one.resample(sample_interval / num_sub_steps).nearest()

        # append each of the sub feature into the data df
        sub_feature_df = pd.DataFrame()
        for i in range(num_sub_steps):
            # the reason to reset index is that sometimes each sub feature's length can differ by 1
            sub_feature_srs = ts_data_one.iloc[i::num_sub_steps].reset_index(drop=True)
            sub_feature_srs = sub_feature_srs.rename("{}_sub_step_{}".format(feature_name, i))
            sub_feature_df = pd.concat([sub_feature_df, sub_feature_srs], axis=1, join="outer")

        ts_data_one = sub_feature_df

    # ==== Option 3 of 3 ====
    # If feature frequency is lower than desired, pad the series into ML frequency
    else:
        ts_data_one = ts_data_one.resample(sample_interval).nearest().to_frame()

    return ts_data_one


def pad_data_w_buffer(ts_data_df, lag_term_configs, lead_term_configs, sample_interval):
    """
    A function to pad the raw data files in both the lag (backwards) and the lead (forwards) direction
    As the lag terms used in input downstream make use of vectorized calculation. A uniform padding allows
    easier manipulation of the data and constant data frame size.

    Args:
        ts_data_df: pd.DataFrame of (M,N). M being number of time points, N being number of features
        lag_term_configs: pd.DataFrame of (I,3). Configuration of lag terms for input features
        lead_term_configs: pd.DataFrame of (O,3). Configuration of lead terms for model responses
        sample_interval: pd.Timedelta. Interval between training/testing samples

    Returns:
        ts_data_df: The feature data frame padded with enough NaNs in lag and lead direction
    """

    # Calculate the maximum amount of lag and lead to determine length of padding
    max_lag_terms = min([lag_term_configs["Start"].min(), 0, lead_term_configs["Start"].min()])
    max_lead_terms = max([lag_term_configs["End"].max(), 0, lead_term_configs["End"].max()])

    # Create padding for lag and lead terms
    lag_terms_time_shift = np.arange(max_lag_terms, 0) * sample_interval
    lead_terms_time_shift = np.arange(1, max_lead_terms + 1) * sample_interval
    lag_pad = pd.DataFrame(index=ts_data_df.index[0] + lag_terms_time_shift, columns=ts_data_df.columns)
    lead_pad = pd.DataFrame(index=ts_data_df.index[-1] + lead_terms_time_shift, columns=ts_data_df.columns)

    # Append the padding to the raw data frame
    padded_ts_data = pd.concat([lag_pad, ts_data_df, lead_pad])

    return padded_ts_data


def generate_lag_and_lead_terms(ts_data_df, lag_term_configs, lead_term_configs):

    # Initialize collectors to hold (and later save) trainval and inf data in
    io_data_df = pd.DataFrame(None, index=ts_data_df.index)
    ts_index = ts_data_df.index
    is_feature_input = pd.Series(None, dtype=bool)
    # Iterate over each feature, including both predictors and responses, to create time shifted input and outputs
    for i, term_configs in enumerate([lag_term_configs, lead_term_configs]):
        is_input = True if i == 0 else False  # all lag terms are considered inputs, while lead_term are outputs
        term_configs = term_configs.astype("int")  # force the integer type as they are basis for range

        for feature_name in term_configs.index:

            # obtain lag term start and end offset, as well as step size for a certain feature
            start, end, step = term_configs.loc[feature_name].iloc[:3]

            # Iterate over each time step for current predictor type
            for time_step in range(start, end + 1, step):
                label = "{}_T{:+}".format(feature_name, time_step)
                io_data_df[label] = None  # initialize feature values
                if time_step > 0:
                    io_data_df.loc[ts_index[:-time_step], label] = ts_data_df.loc[
                        ts_index[time_step:], feature_name
                    ].values
                elif time_step < 0:
                    io_data_df.loc[ts_index[-time_step:], label] = ts_data_df.loc[
                        ts_index[:time_step], feature_name
                    ].values
                else:
                    io_data_df[label] = ts_data_df[feature_name]

                # record if this is an input or output they are inputs
                is_feature_input.loc[label] = is_input

    return io_data_df, is_feature_input


def create_trainval_test_infer_sets(io_data_df, starts_and_ends, is_feature_input, data_dir):
    """
    Takes the combined data set and separates out the trainval, test and inference sets

    Input:
    io_data_df: pd.DataFrame of [M, N2]. M being the number of time points, and N2 being the number of features derived
    from the time series. It holds predictors and response variables, which have all lag and lead terms generated
    starts_and_ends: pd.DataFrame of [M, N2] the start and end defined for the training, testing and inference sets.
    is_feature_input: pd.Series of [N2] bool. A pandas series recording if each feature is an input

    Output:
        None. The created input and output files are directly saved to hard drive

    """

    # Validate training and testing set should be non-overlapping
    if (
        starts_and_ends.loc["test", "Start Time"]
        < starts_and_ends.loc["trainval", "Start Time"]
        < starts_and_ends.loc["test", "End Time"]
    ) or (
        starts_and_ends.loc["test", "Start Time"]
        < starts_and_ends.loc["trainval", "End Time"]
        < starts_and_ends.loc["test", "End Time"]
    ):
        raise ValueError("There is overlap between training and testing data. BAD!")

    # Separate train/inf set from the combined set
    for set_name in starts_and_ends.index:
        set_range = (starts_and_ends.loc[set_name, "Start Time"] <= io_data_df.index) & (
            io_data_df.index < starts_and_ends.loc[set_name, "End Time"]
        )
        set_data_df = io_data_df.loc[set_range].copy()

        # Inference set will not have a response. Delete the response columns
        if set_name == "infer":
            set_data_df = set_data_df.drop(columns=set_data_df.columns[~is_feature_input])

        # summarize data validity and drop invalid samples
        print("{} of {} {} samples are valid".format(set_data_df.dropna().shape[0], set_data_df.shape[0], set_name))
        set_data_df = set_data_df.dropna()

        for input_or_output in ["input", "output"]:
            if (set_name == "infer") and (input_or_output == "output"):
                continue
            else:
                is_looking_for_input = input_or_output == "input"
                # Only retain trainval samples wherein predictor(s) and response(s) are both valid
                set_io_df = set_data_df[io_data_df.columns[is_looking_for_input == is_feature_input]]

                # save to hard drive
                filename = "{}_{}.pkl".format(input_or_output, set_name)
                if len(set_io_df.index) >= 1:  # if there is no data then don't print out anything
                    set_io_df.to_pickle(data_dir / filename)
                else:
                    print("{} for {} set is empty. Please double check!".format(input_or_output, set_name))

    return None


def main(dir_str):

    print("=== Step 1 of 5, Reading in model inputs and time series from {} ===".format(INPUT_EXCEL_NAME))
    configs = ExcelConfigs(INPUT_EXCEL_NAME.resolve())
    # Paths to read time files from. Defined in the dir_structure class in utility
    dir_str = Dir_Structure(model_name= configs.model_name)
    # read in all timeseries files
    ts_data_df = read_all_timeseries(dir_str, configs)

    print("=== Step 2 of 5. Calculating derived features, applies to RESERVE more than RECLAIM ===")
    # configs.lead_term_config.append(pd.DataFrame([1, 2, 2], index=cal_predictors_df))

    print("=== Step 3 of 5, Calculating calendar-based predictors === ")
    cal_predictors = CalendricalPredictors(ts_data_df.index, configs)
    ts_data_df = pd.concat([ts_data_df, cal_predictors.data], axis=1, join="outer")
    configs.lag_term_configs = configs.lag_term_configs.append(cal_predictors.cal_term_configs).astype("int")

    print("=== Step 4 of 5, Using vectorized operations to construct lag and lead terms ===")
    # Pad the raw data with NaNs in both the lag and lead direction for downstream data manipulation
    ts_data_df = pad_data_w_buffer(
        ts_data_df, configs.lag_term_configs, configs.lead_term_configs, configs.sample_interval
    )
    io_data_df, is_feature_input = generate_lag_and_lead_terms(
        ts_data_df, configs.lag_term_configs, configs.lead_term_configs
    )

    print("=== Step 5 of 5. Separate trainval, test and inference sets, and save to hard drive ===")
    create_trainval_test_infer_sets(io_data_df, configs.starts_and_ends, is_feature_input, dir_str.reclaim_data_dir)

    print("All done!")


# run as a script
if __name__ == "__main__":
    main(dir_str)
