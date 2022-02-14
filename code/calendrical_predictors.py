import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
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
            configs:
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
        non_feature_attrs = geo_params + [
            "dt_series",
            "data",
        ]  # the attributes that don't end up as features

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

        # generation the lag configuration for the Calendar terms
        self.cal_term_configs = pd.DataFrame(
            1, index=self.data.columns, columns=configs.lag_term_configs.columns
        )
        self.cal_term_configs *= [
            0,
            0,
            1,
        ]  # no reason to include lagged version of the Calendar terms

    def get_holiday(self):
        """
        See if the datetime falls on a holiday
        """
        holidays = Calendar().holidays(
            start=self.dt_series.min(), end=self.dt_series.max()
        )
        self.is_holiday = pd.Series(
            self.dt_series.isin(holidays), index=self.dt_series
        ).astype("int")

    def get_day_of_week(self):
        # Day of week indicator. 7 binary indicator variables, one each for Mon to Sun
        day_of_week = self.dt_series.day_name()
        self.is_day_of_week = pd.get_dummies(day_of_week, prefix="is").set_index(
            self.dt_series
        )

    def get_revolution_angle(self):
        """
        Revolution angle corresponds to day of year, transformed to radians and in turn sine and cosine.
        """
        # The grammar here is a bit hacky, but the second subtraction sign is overloaded with pd syntax to move dates
        time_from_year_start = self.dt_series - (
            self.dt_series - pd.offsets.YearBegin()
        )
        time_in_a_year = (self.dt_series + pd.offsets.YearBegin()) - (
            self.dt_series - pd.offsets.YearBegin()
        )

        # calculate revolution angle and sine and cosine values
        rev_angle = time_from_year_start / time_in_a_year * 2 * np.pi
        self.sin_rev_angle = pd.Series(np.sin(rev_angle), index=self.dt_series)
        self.cos_rev_angle = pd.Series(np.cos(rev_angle), index=self.dt_series)

    def get_rotation_angle(self):
        """
        Rotation angle corresponds to hour of day, transformed to radians and in turn trigonometric values.
        """
        time_from_day_start = self.dt_series - self.dt_series.normalize()
        rot_angle = time_from_day_start / pd.Timedelta("1D") * 2 * np.pi
        self.sin_rot_angle = pd.Series(np.sin(rot_angle), index=self.dt_series)
        self.cos_rot_angle = pd.Series(np.cos(rot_angle), index=self.dt_series)

    def get_elapsed_time(self):
        """
        Representing the temporal sequence of all the time points, especially important if there is growth in
        installation or load. Represented as days or fraction of days since an arbitrarily defined start_date

        """
        self.elapsed_time = (self.dt_series - START_DATE).total_seconds() / 3600 / 24
        self.elapsed_time = pd.Series(self.elapsed_time, index=self.dt_series)

    def get_solar_position(self):
        UTC_time = self.dt_series - self.tz_from_utc * pd.Timedelta("1H")
        position_df = pvlib.solarposition.get_solarposition(
            UTC_time, self.latitude, self.longitude
        )
        # Convert back to local time and to degrees
        position_df = np.deg2rad(position_df.set_index(self.dt_series))
        self.sin_solar_zenith_angle = np.sin(position_df["apparent_zenith"])
        self.cos_solar_zenith_angle = np.cos(position_df["apparent_zenith"])
        self.sin_solar_azimuth_angle = np.sin(position_df["azimuth"])
        self.cos_solar_azimuth_angle = np.cos(position_df["azimuth"])


def calculate_clear_sky_output(
    datetime_arr, latitude, longitude, time_difference_from_UTC
):
    """

    Args:
        datetime_arr(pd.DatetimeIndex)
        latitude(float): Latitude to be used to calculate solar elevation
        longitude(float): Longitude to be used to calculate local solar time in degrees. East->positive, West->Negative
        time_difference_from_UTC(int/float): Time-difference (in hours) between local time and
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
    solar_declination_arr = -23.5 * np.cos(
        np.deg2rad((360 / 365) * (day_of_year_arr + 10))
    )  # degrees
    solar_declination_in_radians_arr = np.deg2rad(solar_declination_arr)  # radians

    # Calculate solar altitude in each period
    latitude_in_radians = np.deg2rad(latitude)  # radians
    solar_elevation_angle_in_radians_arr = np.arcsin(
        np.sin(solar_declination_in_radians_arr) * np.sin(latitude_in_radians)
        + np.cos(solar_declination_in_radians_arr)
        * np.cos(latitude_in_radians)
        * np.cos(solar_hour_angle_in_radians_arr)
    )  # radians

    # Calculate normalized clear sky output (proportional to sin(solar elevation angle))
    clear_sky_output_arr = np.sin(solar_elevation_angle_in_radians_arr)  # W/m^2
    # clear_sky_output cannot be negative; correct negative values to 0
    zeros = (
        np.zeros(len(clear_sky_output_arr)) + 0.001
    )  # Small constant added to avoid divide by zero errors
    clear_sky_output_arr = np.max([clear_sky_output_arr, zeros], axis=0)

    # Create clear_sky_output dataframe
    clear_sky_output_df = pd.DataFrame(
        index=datetime_arr,
        columns=["clear_sky_output"],
        data=np.array(clear_sky_output_arr),
    )

    return clear_sky_output_df
