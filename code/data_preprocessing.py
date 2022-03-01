import pandas as pd
import pathlib
from utility import DirStructure
from parse_excel_configs import ExcelConfigs
from calendrical_predictors import CalendricalPredictors
from data_preprocessing_util import *

# ==== Constants  ====
# Input excel name
INPUT_EXCEL_NAME = pathlib.Path("RESERVE_input_v1.xlsx")


def main():

    print("=== Step 1 of 5, Parse model configs from {} ===".format(INPUT_EXCEL_NAME))
    configs = ExcelConfigs(INPUT_EXCEL_NAME.resolve())
    # Paths to read time files from. Defined in the dir_structure class in utility
    dir_str = DirStructure(model_name=configs.model_name)

    print("=== Step 2.1 of 5 synthesize forecast series when needed===")
    configs = synthesize_forecast(configs, dir_str)

    print("=== Step 2.2 of 5.Read in all timeseries ===")
    # read in all timeseries files
    ts_data_df, sub_ts_dict = read_all_timeseries(dir_str, configs)
    # calculate response variables
    # TODO: Add a check if no time series is defined as output and no response variable has been calculated
    print("=== step 2.3 of 5 calculate forecast errors when required ===")
    # This applies to RESERVE more than RECLAIM and is only required some but not at all times
    if configs.synthesize_forecast_error:
        fc_err_df = calculate_forecast_error(ts_data_df, configs, dir_str)
        ts_data_df = pd.concat([ts_data_df, fc_err_df], axis=1, join="outer")

    # Replace timeseries with sub timeseries, applicable to down-sampled ts
    ts_data_df = concat_sub_ts(ts_data_df, sub_ts_dict, configs)

    print("=== Step 3 of 5, Calculating Calendar-based predictors === ")
    cal_predictors = CalendricalPredictors(ts_data_df.index, configs)
    ts_data_df = pd.concat([ts_data_df, cal_predictors.data], axis=1, join="outer")
    configs.lag_term_configs = pd.concat(
        [configs.lag_term_configs, cal_predictors.cal_term_configs]
    )

    print("=== Step 4 of 5, Vectorized construction of lag and lead terms ===")
    # Pad the raw data with NaNs in both the lag and lead direction for downstream data manipulation
    ts_data_df = pad_data_w_buffer(
        ts_data_df,
        configs.lag_term_configs,
        configs.lead_term_configs,
        configs.sample_interval,
    )
    io_data_df, is_feature_input = generate_lag_and_lead_terms(
        ts_data_df, configs.lag_term_configs, configs.lead_term_configs
    )

    print("=== Step 5 of 5. Separate trainval, test and inference sets and save ===")
    create_trainval_test_infer_sets(
        io_data_df, configs.starts_and_ends, is_feature_input, dir_str.data_dir
    )

    print("All done!")


# run as a script
if __name__ == "__main__":
    main()
