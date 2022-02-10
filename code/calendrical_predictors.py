import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
import pvlib

# used in the elapsed time attribute calculation
START_DATE = pd.Timestamp("20200101")


class CalendricalPredictors(object):
    def __init__(self, dt_series, configs):
        """
        Reference for formula:C.B.Honsberg and S.G.Bowden, “Photovoltaics Education Website,” www.pveducation.org, 2019
        Args:
            dt_series:
            lat:
            long:
        """
        geo_params = ["latitude", "longitude", "tz_from_utc"]
        for geo_param in geo_params:
            if hasattr(configs, geo_param):
                setattr(self, geo_param, getattr(configs, geo_param))
            else:
                setattr(self, geo_param, 0)

        # Initialization of all class attributes
        self.dt_series = pd.to_datetime(dt_series)
        self.data = pd.DataFrame(index=dt_series)
        self.is_holiday = None
        self.is_day_of_week = None
        self.elapsed_time = None
        self.sin_rev_angle = None
        self.cos_rev_angle = None
        self.sin_rot_angle = None
        self.cos_rot_angle = None
        self.sin_solar_azimuth_angle = None
        self.cos_solar_azimuth_angle = None
        self.sin_solar_zenith_angle = None
        self.cos_solar_zenith_angle = None
        non_feature_attrs = geo_params + ["dt_series", "data"]  # the attributes that doesn't end up being features

        for temporal_feature in configs.temporal_features.index:
            if configs.temporal_features.loc[temporal_feature, "To include?"]:
                getattr(self, "get_" + temporal_feature.lower().replace(" ", "_"))()

        # collect features into the data attribute:
        for attr in self.__dict__.keys():
            attribute_data = getattr(self, attr)
            if (attr not in non_feature_attrs) and (attribute_data is not None):
                if isinstance(attribute_data, pd.Series):
                    self.data[attr] = attribute_data
                elif isinstance(attribute_data, pd.DataFrame):
                    self.data = pd.concat([self.data, attribute_data], axis=1)

        # generation the lag configuration for the calendar terms
        self.cal_term_configs = pd.DataFrame(1, index=self.data.columns, columns=configs.lag_term_configs.columns)
        self.cal_term_configs *= [0, 0, 1]  # no reason to include lagged version of the calendar terms

    def get_holiday(self):
        """
        See if the datetime falls on a holiday
        """
        holidays = calendar().holidays(start=self.dt_series.min(), end=self.dt_series.max())
        self.is_holiday = self.dt_series.isin(holidays)

    def get_day_of_week(self):
        # Day of week indicator. 7 binary indicator variables, one each for Mon to Sun
        day_of_week = self.dt_series.day_name()
        self.is_day_of_week = pd.get_dummies(day_of_week, prefix="is").set_index(self.dt_series)

    def get_revolution_angle(self):
        """
        Revolution angle corresponds to day of year, transformed to radians and in turn sine and cosine.
        """
        # The grammar here is a bit hacky, but the second subtraction sign is overloaded with pd syntax to move dates
        time_from_year_start = self.dt_series - (self.dt_series - pd.offsets.YearBegin())
        time_in_a_year = (self.dt_series + pd.offsets.YearBegin()) - (self.dt_series - pd.offsets.YearBegin())

        # calculate revolution angle and sine and cosine values
        rev_angle = time_from_year_start / time_in_a_year * 2 * np.pi
        self.sin_rev_angle = np.sin(rev_angle)
        self.cos_rev_angle = np.cos(rev_angle)

    def get_rotation_angle(self):
        """
        Rotation angle corresponds to hour of day, transformed to radians and in turn trigonometric values.
        """
        time_from_day_start = self.dt_series - self.dt_series.normalize()
        rot_angle = time_from_day_start / pd.Timedelta("1D") * 2 * np.pi
        self.sin_rot_angle = np.sin(rot_angle)
        self.cos_rot_angle = np.cos(rot_angle)

    def get_elapsed_time(self):
        """
        Representing the temporal sequence of all the time points, especially important if there is growth in installation
        or load. Represented as days or fraction of days since an arbitrarily defined start_date

        """
        self.elapsed_time = (self.dt_series - START_DATE).total_seconds() / 3600 / 24

    def get_solar_position(self):
        position_df = pvlib.solarposition.get_solarposition(self.dt_series, self.latitude, self.longitude)
        self.sin_solar_zenith_angle = np.sin(position_df["apparent_zenith"])
        self.cos_solar_zenith_angle = np.cos(position_df["apparent_zenith"])
        self.sin_solar_azimuth_angle = np.sin(position_df["azimuth"])
        self.cos_solar_azimuth_angle = np.cos(position_df["azimuth"])
