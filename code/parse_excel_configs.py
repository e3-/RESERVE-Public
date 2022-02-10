import pandas as pd
import xlwings as xw


class ExcelConfigs(object):
    """
    The object of this class would read and contain input information defined in a excel workbook.

    Some of the tabs to expect here are presented below, while more can be defined by the user:
    ts_attrs: pd.DataFrame of (N,?) strs. Record the file name and variable name for each time series
    file. N being the number of timeseries files.
    sets_starts_and_ends: pd.DataFrame of (3,2) pd.Timestamps. The start and end date of the data sets
    lag_term_configs: pd.DataFrame of (I,3) ints. The start, end and step of the lag terms.
    I being the number of input timeseries.See more instructions in the input excel workbook
    lead_term_configs: pd.DataFrame of (O,3) ints. The start, end and step of the lead terms.
    O being the number of output timeseries. See more instructions in the input excel workbook
    main_parameters:pd.DataFrame of (M,?) mixed. Collects various main parameters for inputs.
        In main parameters. it is often essential to have model name and samples interval


    """

    essential_params = [
        "model_name",
        "sample_interval",
        "timeseries_attributes",
        "starts_and_ends",
        "lag_term_configs",
        "lead_term_configs",
        "temporal_features",
        "main_parameters",
    ]

    set_names_to_shorthand = {
        "Training and Validation Set": "trainval",
        "Testing Set": "test",
        "Inference Set": "infer",
    }

    def __init__(self, input_excel_path):
        """
        While initializing the input object based on the excel file, it will subject to the three following treatments
        and should be designed with these in mind
        1. Every tab will be transformed into a pd DataFrame spanning cell "A1" to the the first empty row/column
        2. The data frame will become an attribute of the object, and will have a name based on the tab. All lower cases with spaces substituted with _
        3. The initializer will attempt to find a tab called "Main Params", any parameters defined in this tab will be exposed and expanded into an attribute.
        Args:
            input_excel_path: pathlib.Path defining the path to the input excel file
        """

        # Connect to the reclaim input workbook
        reclaim_input_wb = xw.Book(input_excel_path)

        # Loop thru each of the tabs and read them all in as different pd Dataframe
        for sheet in reclaim_input_wb.sheets:
            df = sheet.range("A1").options(pd.DataFrame, expand="table").value
            attr_name = sheet.name.lower().replace(" ", "_")
            self.__setattr__(attr_name, df)

        # Find the main params
        if "main_parameters" in self.__dict__.keys():
            # list each row of the main parameters tab as an attribute
            for param_name in self.main_parameters.index:
                attr_name = param_name.lower().replace(" ", "_")
                self.__setattr__(attr_name, self.main_parameters.loc[param_name, "Value"])
        else:
            print("main paramters tab not found, highly recommended to define one")

        for param_name in self.essential_params:
            if param_name not in self.__dict__.keys():
                raise ValueError(param_name + " not found, essential for model run!")

        # parse sample interval into pd.Timedelta
        self.sample_interval = pd.Timedelta(self.sample_interval)

        # parse the data set's names into short hands:
        self.starts_and_ends.rename(index=self.set_names_to_shorthand, inplace=True)
