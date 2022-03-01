import os
import numpy as np
import pandas as pd
import pvlib

# Column names used in data-checker output files
COL_NAME_VALUE = "Value_Interval_Avg"
COL_NAME_VALIDITY = "valid_all_checks"
COL_NAME_DATETIME = "Datetime_Interval_Start"

# Relationship between lag, lead, input and output
io_lag_lead_map = {"input": "lag", "output": "lead"}


def synthesize_forecast(configs, dir_str):
    """
    Synthesize forecast of certain timeseries, which almost always comes from persistence
    :param configs: parse_excel_configs.ExcelConfig. Configuration of the data
    preprocessing procedure
    :param dir_str: utility.DirStructure. Directory structure of the current model
    :return: None as files are saved to hard drive
    """

    fc_configs = configs.forecast_configs  # alias
    fc_contrib = configs.forecast_error_contribution  # alias
    ts_attrs = configs.timeseries_attributes  # alias

    # loop through all the timeseries in the forecast configs tab
    for ts_name in fc_configs.index:

        if fc_configs.loc[ts_name, "Synthesize Forecast?"]:
            # TODO: if forecast exists then skip this process
            print("... Synthesizing forecast for {}...".format(ts_name))
            # extract lead time and convert it to amount of lead term
            forecast_horizon = pd.Timedelta(fc_configs.loc[ts_name, "Forecast Horizon"])
            ts_csv_df = pd.read_csv(
                os.path.join(
                    dir_str.data_checker_dir, ts_attrs.loc[ts_name, "File Name"]
                ),
                index_col=COL_NAME_DATETIME,
                parse_dates=True,
                infer_datetime_format=True,
            )

            if fc_configs.loc[ts_name, "Method"] == "persistence":
                # For the persistence forecast, forecast for T + forecast_lead_time
                # is the value of that timeseries at T time
                ts_forecast = ts_csv_df.set_index(ts_csv_df.index + forecast_horizon)

            elif fc_configs.loc[ts_name, "Method"] == "solar persistence":
                # Slightly more complicated than the persistence method, it assumes that
                # the cloudiness (solar output/ clear sky output) would remain the same
                zenith_fc = pvlib.solarposition.get_solarposition(
                    ts_csv_df.index
                    + forecast_horizon
                    - configs.tz_from_utc * pd.Timedelta("1h"),
                    configs.latitude,
                    configs.longitude,
                )["apparent_zenith"]
                zenith = pvlib.solarposition.get_solarposition(
                    ts_csv_df.index - configs.tz_from_utc * pd.Timedelta("1h"),
                    configs.latitude,
                    configs.longitude,
                )["apparent_zenith"]

                cos_zenith_ratio = np.cos(np.deg2rad(zenith_fc.values)) / np.cos(
                    np.deg2rad(zenith.values)
                )
                # Cap adjustment factors at 2
                cos_zenith_ratio = np.clip(cos_zenith_ratio, a_min=0, a_max=2)
                ts_forecast = ts_csv_df.set_index(ts_csv_df.index + forecast_horizon)
                ts_forecast[COL_NAME_VALUE] *= cos_zenith_ratio

            else:
                raise ValueError(
                    "Only persistence and solar persistence are supported!"
                )

            # Save forecast
            forecast_filename = "{}_forecast_T+{:.0f}min.csv".format(
                ts_name, forecast_horizon / pd.Timedelta("1T")
            )
            ts_forecast.to_csv(
                os.path.join(dir_str.data_checker_dir, forecast_filename)
            )

            # Add the ts forecast to the timeseries.
            ts_fc_name = ts_name + "_forecast"
            fc_contrib.loc[ts_fc_name] = fc_contrib.loc[ts_name]
            fc_contrib.loc[ts_fc_name, "Forecast or Actual"] = "Forecast"
            ts_attrs.loc[ts_fc_name, "File Name"] = forecast_filename
            ts_attrs.loc[ts_fc_name, ["Is Input?", "Is Output?"]] = [True, False]

            # Generate lag terms configs for it
            configs.lag_term_configs.loc[ts_fc_name] = fc_configs.loc[
                ts_name,
                ["Forecast Term Start", "Forecast Term End", "Forecast Term Step"],
            ].values

    return configs


def read_all_timeseries(dir_str, configs):
    """
    Read in all time series according to the timeseries attribute tab
    :param configs: parse_excel_configs.ExcelConfig. Configuration of the data preprocessing procedure
    :param dir_str: utility.DirStructure. Directory structure of the current model
    :return: ts_data_df: containing the timeseries information, with 1 column corresponding to one row in the input
    sub_ts_dict: dict(str: pd.DataFrame). For timeseries that are at least twice as more frequent than sample interval,
    sub time series would be created to preserve this information.
    """

    ts_attrs = configs.timeseries_attributes  # alias

    # Reading in time series and matching them to the frequency needed
    ts_data_df = pd.DataFrame()  # empty container for all time series and
    sub_ts_dict = {}  # empty container for all sub-time series
    # Iterate over each time series from data-checker
    for ts_name in ts_attrs.index:

        print("Reading in " + ts_name + "...")
        ts_csv_df = pd.read_csv(
            os.path.join(dir_str.data_checker_dir, ts_attrs.loc[ts_name, "File Name"]),
            index_col=COL_NAME_DATETIME,
            parse_dates=True,
            infer_datetime_format=True,
        )
        # match the feature frequency with the required frequency of ML problem
        ts_data_one, sub_ts_dict[ts_name] = match_frequency(
            ts_csv_df, ts_name, configs.sample_interval
        )

        # collect each individual feature or sub-feature into the total timeseries data dataframe
        ts_data_df = pd.concat([ts_data_df, ts_data_one], axis=1, join="outer")

    return ts_data_df, sub_ts_dict


def match_frequency(ts_csv_df, ts_name, sample_interval):
    """
    Unstack or pad the original ts in order to achieve desired frequency.

    Args:
        ts_csv_df: pd.DataFrame. the dataframe of the ts read in from hard drive, following data checker output format
        ts_name: str. Name of the ts
        sample_interval: int. The time step of the ML model expressed in seconds.

    Returns:
        ts_data_srs: pd.Series of pd.DataFrame. Time series data

    """

    # Embed info about validity, so we can use the ts_data_df alone going forward
    ts_csv_df.loc[~ts_csv_df[COL_NAME_VALIDITY], COL_NAME_VALUE] = None
    # sub timeseries are sometimes generated from frequency matching of high freq ts
    sub_ts_df = None
    # Rename from generic col name to ts-specific name
    ts_data_one = ts_csv_df[COL_NAME_VALUE].rename(ts_name)

    # Determine time-step size in the data for this ts
    ts_freq = pd.infer_freq(ts_data_one.index)
    assert (
        ts_freq is not None
    ), "Raw data does not have equally spaced index! Cannot discern Frequency!"

    # If inferred frequency is 1 of something, say 1 min or 1 H, it will just be represented
    # as "T" or "H". Need to turn it into "1T" or "1H" to interpret as a number using the Timedelta function
    if ts_freq[0].isdigit():
        ts_time_step = pd.Timedelta(ts_freq)
    else:
        ts_time_step = pd.Timedelta("1" + ts_freq)

    # TODO: The script takes care of frequency mismatch, but not starting time mismatch.

    # ==== Option 1 of 3 ====
    # If the ts time-step size matches, place the ts into the ML inputs df without any changes
    if ts_time_step == sample_interval:
        ts_data_one = ts_data_one.to_frame()

    # ==== Option 2 of 3 ====
    # If the input is more frequent than desired, create multiple timeseries
    # E.g. Given 5-min ts and ML inputs are on a 15-min resolution. Then, create three 15-minutely tss
    # out of each one of the 5-min ts currently being assessed
    elif ts_time_step < sample_interval:
        # calculate number of sub steps and organize ts frequency into sth that can be obtained
        # through dividing ML time step with an integer
        num_sub_steps = sample_interval // ts_time_step
        ts_data_one_resampled = ts_data_one.resample(
            sample_interval / num_sub_steps
        ).nearest()

        # pick the average, when you need to use a single ts instead of all the sub ts
        ts_data_one = ts_data_one.resample(sample_interval).mean().to_frame()
        sub_ts_df = pd.DataFrame(index=ts_data_one.index)

        # append each of the sub ts into the data df
        for i in range(num_sub_steps):
            # set index to have the same frequency as sample interval
            sub_ts_df["{}_sub_step_{}".format(ts_name, i)] = ts_data_one_resampled.iloc[
                i::num_sub_steps
            ].values

    # ==== Option 3 of 3 ====
    # If ts frequency is lower than desired, pad the series into ML frequency
    else:
        ts_data_one = ts_data_one.resample(sample_interval).nearest().to_frame()

    return ts_data_one, sub_ts_df


def calculate_forecast_error(ts_data_df, configs, dir_str):
    """
    Calculates and stores response variable(s) that the ML model will be trained to
    predict
    :param ts_data_df: pd.DataFrame containing all predictors. Some if not all of them
    will be used to calculate response(s)
    :param configs: parse_excel_configs.ExcelConfig. Configuration of the data preprocessing procedure
    :param dir_str: utility.DirStructure. Directory structure of the current model
    :return: ts_data_df: pd.DataFrame of (M,N+R). M being the number of data points, N being the
    original number of time series, while R being the newly generated forecast error time series
    equal to the number of timeseries category +1
    """
    # Initialize df to store response variable(s)
    fe_contrib = configs.forecast_error_contribution
    fe_configs = configs.forecast_error_configs

    # when actual load is higher than forecast or when forecast gen is higher than
    # actual, you need upward reserve
    is_upward_reserve = (fe_contrib["Generation or Load"] == "Load") == (
        fe_contrib["Forecast or Actual"] == "Actual"
    )
    fc_err_multipliers = (is_upward_reserve - 0.5) * 2  # convert (0,1) to (-1,1)
    fc_err_df = pd.DataFrame(index=ts_data_df.index)
    for fe_cat in fe_configs.index:
        if fe_cat == "Net Load Forecast Error":
            continue

        if fe_configs.loc[fe_cat, "Synthesize Error?"]:
            fc_error_name = fe_cat + "_Forecast_Error"
            mask_of_category = (fe_contrib["Category"] == fe_cat) & (
                fe_contrib["Impacts Forecast Error?"]
            )
            # pick out the timeseries of this category, and assign positive and negative
            # based on forecast/actual
            ts_fc_err = (
                ts_data_df[ts_data_df.columns[mask_of_category]]
                * fc_err_multipliers[mask_of_category].values
            )
            # No NaN is allowed in the summation process. Any NaN would nullify the
            # entire row
            fc_err_df[fc_error_name] = ts_fc_err.sum(
                min_count=mask_of_category.sum(), axis=1
            )
            configs.lead_term_configs.loc[fc_error_name] = fe_configs.loc[
                fe_cat,
                [
                    "Error Lead Term Start",
                    "Error Lead Term End",
                    "Error Lead Term Step",
                ],
            ].values

    # save to hard drive
    # Calculate net load forecast error by adding this category's forecast error
    if fe_configs.loc["Net Load Forecast Error", "Synthesize Error?"]:
        fc_err_df["Net_Load_Forecast_Error"] = fc_err_df.sum(
            axis=1, min_count=fc_err_df.shape[1]
        )
        configs.lead_term_configs.loc["Net_Load_Forecast_Error"] = fe_configs.loc[
            fe_cat,
            [
                "Error Lead Term Start",
                "Error Lead Term End",
                "Error Lead Term Step",
            ],
        ].values
        fc_err_df.to_csv(os.path.join(dir_str.data_checker_dir, "total_fc_error.csv"))

    return fc_err_df


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
    max_lag_terms = min(
        [lag_term_configs["Start"].min(), 0, lead_term_configs["Start"].min()]
    )
    max_lead_terms = max(
        [lag_term_configs["End"].max(), 0, lead_term_configs["End"].max()]
    )

    # Create padding for lag and lead terms
    lag_terms_time_shift = np.arange(max_lag_terms, 0) * sample_interval
    lead_terms_time_shift = np.arange(1, max_lead_terms + 1) * sample_interval
    lag_pad = pd.DataFrame(
        index=ts_data_df.index[0] + lag_terms_time_shift, columns=ts_data_df.columns
    )
    lead_pad = pd.DataFrame(
        index=ts_data_df.index[-1] + lead_terms_time_shift, columns=ts_data_df.columns
    )

    # Concatenate the padding to the raw data frame
    padded_ts_data = pd.concat([lag_pad, ts_data_df, lead_pad])

    return padded_ts_data


def generate_lag_and_lead_terms(ts_data_df, lag_term_configs, lead_term_configs):

    # Initialize collectors to hold (and later save) trainval and inf data in
    io_data_df = pd.DataFrame(None, index=ts_data_df.index)
    ts_index = ts_data_df.index
    is_feature_input = pd.Series(None, dtype=bool)
    # Iterate over each feature, including both predictors and responses, to create time shifted input and outputs
    for i, term_configs in enumerate([lag_term_configs, lead_term_configs]):
        is_input = (
            True if i == 0 else False
        )  # all lag terms are considered inputs, while lead_term are outputs
        term_configs = term_configs.astype(
            "int"
        )  # force the integer type as they are basis for range

        for feature_name in term_configs.index:

            # obtain lag term start and end offset, as well as step size for a certain feature
            start, end, step = term_configs.loc[feature_name, ["Start", "End", "Step"]]

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


def create_trainval_test_infer_sets(
    io_data_df, starts_and_ends, is_feature_input, data_dir
):
    """
    Takes the combined data set and separates out the trainval, test and inference sets

    Input:
    io_data_df: pd.DataFrame of [M, N2]. M being the number of time points, and N2 being the number of features derived
    from the time series. It holds predictors and response variables, which have all lag and lead terms generated
    starts_and_ends: pd.DataFrame of [M, N2] the start and end defined for the training, testing and inference sets.
    is_feature_input: pd.Series of [N2] bool. A recording of whether each feature is an input

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
        set_range = (
            starts_and_ends.loc[set_name, "Start Time"] <= io_data_df.index
        ) & (io_data_df.index < starts_and_ends.loc[set_name, "End Time"])
        set_data_df = io_data_df.loc[set_range].copy()

        # Inference set will not have a response. Delete the response columns
        if set_name == "infer":
            set_data_df = set_data_df.drop(
                columns=set_data_df.columns[~is_feature_input]
            )

        # summarize data validity and drop invalid samples
        print(
            "{} of {} {} samples are valid".format(
                set_data_df.dropna().shape[0], set_data_df.shape[0], set_name
            )
        )
        set_data_df = set_data_df.dropna()

        for input_or_output in ["input", "output"]:
            if (set_name == "infer") and (input_or_output == "output"):
                continue
            else:
                is_looking_for_input = input_or_output == "input"
                # Only retain trainval samples wherein predictor(s) and response(s) are both valid
                set_io_df = set_data_df[
                    io_data_df.columns[is_looking_for_input == is_feature_input]
                ]

                # save to hard drive
                filename = "{}_{}.pkl".format(input_or_output, set_name)
                if (
                    len(set_io_df.index) >= 1
                ):  # if there is no data then don't print out anything
                    set_io_df.astype("float32").to_pickle(data_dir / filename)
                else:
                    print(
                        "{} for {} set is empty. Please double check!".format(
                            input_or_output, set_name
                        )
                    )

    return None


def concat_sub_ts(ts_data_df, sub_ts_dict, configs):
    """
    Combine the sub time series into previously generated timeseries DF
    :param ts_data_df: pd.DataFrame of (M, N+R). At this point still not including any sub time series
    :param sub_ts_dict: dict[str, pd.DataFrame]. The name of the timeseries as key, and unstacked dataframe as values
    :param configs: parse_excel_configs.ExcelConfigs. Configuration file of how to run this script
    :return: ts_data_df: pd.DataFrame of (M, N + R +S) S being the net increase in timeseries when you count
    sub time series.
    """

    ts_attrs = configs.timeseries_attributes
    for ts_name, sub_ts_df in sub_ts_dict.items():
        if sub_ts_df is not None:  # skip all timeseries with no sub
            # Replace the 1 column timeseries with the multi-column sub timeseries
            ts_data_df.drop(columns=ts_name, inplace=True)
            ts_data_df = pd.concat((ts_data_df, sub_ts_df), axis=1, join="outer")

            # The match frequency process sometimes change the name and amount of time series
            for data_cat, term_cat in io_lag_lead_map.items():
                col_to_check = "Is {}?".format(data_cat.capitalize())
                term_configs_name = "{}_term_configs".format(term_cat)
                term_configs = getattr(configs, term_configs_name)

                if ts_attrs.loc[ts_name, col_to_check]:
                    addl_term_configs = pd.DataFrame(
                        1, index=sub_ts_df.columns, columns=term_configs.columns
                    )
                    addl_term_configs *= term_configs.loc[ts_name].values
                    # add in the new and drop the old
                    term_configs = pd.concat([term_configs, addl_term_configs]).drop(
                        ts_name
                    )
                    setattr(configs, term_configs_name, term_configs)

    return ts_data_df
