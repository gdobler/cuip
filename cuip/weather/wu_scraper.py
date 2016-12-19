from __future__ import print_function
import argparse
import os
import sys
import datetime
import multiprocessing
import itertools
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from cuip.cuip.utils import cuiplogger


class Weather(object):

    def __init__(self):
        """
        Parameters
        ----------
        pwsid: str
            weather underground's station id
        """
        self.station_url = "http://www.wunderground.com/weatherstation/WXDailyHistory.asp?ID={pwsid}&day={day}&month={month}&year={year}&graphspan=day&format=1"
        self.airport_url = "https://www.wunderground.com/history/airport/{airport}/{year}/{month}/{day}/DailyHistory.html?format=1"
        self.logger = cuiplogger.cuipLogger(loggername="wu_scraper", tofile=False)

    def from_airport(self, year, month, day, airport=None):
        """
        Grab the weather data from weather underground's airport based
        weather stations
        
        Parameters
        ----------
        year: int
        month: int
        day: int
        airport: str
            airport code prefixed by 'k'. example 'kjfk'
        """
        self.logger.info("Obtaining airport weather data for {year}-{month}-{day}".\
                             format(year=year, month=month, day=day))
        # Create urls
        airport_url  = self.airport_url.format(airport=airport, year=year, month=month, day=day)
        # weather underground keeps changing Airport time column name as one of the following
        air_time_cols = ["TimeEDT", "TimeEST", "Time"]

        def airport_dateparser(dt):
            """
            Parse column with date and time from airport weather information
            """
            hh = datetime.datetime.strptime(dt, '%I:%M %p').strftime('%H')
            mm = datetime.datetime.strptime(dt, '%I:%M %p').strftime('%M')
            return datetime.datetime.combine(datetime.date(year, month, day), 
                                             datetime.time(int(hh), int(mm)))
        try:
            airport_cols = pd.read_csv(airport_url, nrows=1).columns
            time_col_name = set(air_time_cols).intersection(airport_cols).pop()
            airport_data = pd.read_csv(airport_url, names=airport_cols[:-1], 
                                       usecols=airport_cols[:-1], parse_dates=[time_col_name], 
                                       infer_datetime_format=False, date_parser=airport_dateparser, 
                                       skiprows=2)
            # convert string timestamp to datetime
            airport_data.rename(columns={time_col_name: 'Time'}, inplace=True)
            airport_data['Time'] = pd.to_datetime(airport_data['Time'])
            return airport_data
        except Exception as ex:
            self.logger.error("Error obtaining weather information from airport for: "+
                              "-".join([year, month, day])+
                              " "+str(ex))

    def from_weather_station(self, year, month, day, pwsid=None):
        """
        Grab the weather data from Weather Underground personal weather stations
        
        Parameters
        ----------
        year : int
        month : int
        day : int
        pwsid : str
        weather underground's personal weather station id
        example "KNYNEWYO116" for BoreumHill NYC
        Returns
        -------
        Pandas DataFrame containing weather data for particular date.
        """
        self.logger.info("Obtaining station weather data for {year}-{month}-{day}".\
                             format(year=year, month=month, day=day))
        # Create urls
        station_url  = self.station_url.format(pwsid=pwsid, year=year, month=month, day=day)
        
        try:
            # fetch weather data
            station_cols = pd.read_csv(station_url, nrows=1).columns                                                                                                   
            station_data = pd.read_csv(station_url, names=station_cols[:-1], 
                                       usecols=station_cols[:-1], parse_dates=["Time"], 
                                       infer_datetime_format=True)
            # convert string timestamp to datetime
            station_data = station_data[1::2]
            station_data['Time'] = pd.to_datetime(station_data['Time'])
            return station_data
        except Exception as ex:
            self.logger.error("Error obtaining weather information from pws for: "+
                              "-".join([year, month, day])+
                              " "+str(ex))

    def remove_dups(self, df, tablename, engine, dup_cols=[]):
        """
        Remove rows from a dataframe that already exist in a database
        https://github.com/ryanbaumann/Pandas-to_sql-upsert/blob/master/to_sql_newrows.py
        Parameters
        ----------
        df: pandas.DataFrame
            dataframe to remove duplicate rows from
        engine: SQLAlchemy engine object
        tablename: str
            tablename to check duplicates in
        dup_cols: list or tuple 
            column name(s) to check for duplicate row values
        Returns
        -------
        Unique list of values from dataframe compared to database table
        """
        args = 'SELECT %s FROM %s' %(', '.join(['"{0}"'.format(col) for col in dup_cols]), tablename)
        df.drop_duplicates(dup_cols, keep='last', inplace=True)
        df = pd.merge(df, pd.read_sql(args, engine), how='left', on=dup_cols, indicator=True)
        df = df[df['_merge'] == 'left_only']
        self.logger.info("Dropping duplicates")
        df.drop(['_merge'], axis=1, inplace=True)
        # convert certain columns to numberic datatype so that when passing
        # to sql database, there are no errors
        numeric_cols = ["TemperatureF", "Dew PointF", "Humidity", 
                        "Sea Level PressureIn", "VisibilityMPH", 
                        "Wind SpeedMPH", "Gust SpeedMPH", 
                        "PrecipitationIn", "WindDirDegrees"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def to_database(self, dbname=None, tablename=None, dataframe=None):
        """
        Add dataframe to database
        """
        engine = create_engine('postgresql:///%s'%dbname)
        conn = engine.connect()
        self.logger.info("Total {tot} cols".format(tot=dataframe.shape[0]))
        def _initialize_table(conn, tablename=None):
            """
            Create table schema and commit
            Parameters
            ----------
            conn: sql alchemy connection
            tablename: name of the weather table
            ..note: If the table exists, it will not perform any action
            """
            # create table if it doesn't exist
            if not conn.execute("select exists(SELECT * FROM information_schema.tables WHERE table_name=%s)", 
                                (tablename,)).fetchone()[0]:
                self.logger.warning("Creating New Table")
                conn.execute('CREATE TABLE weather \
                              ("Time" timestamp without time zone PRIMARY KEY NOT NULL, \
                              "TemperatureF" REAL, \
                              "Dew PointF" REAL, \
                              "Humidity" REAL, \
                              "Sea Level PressureIn" REAL, \
                              "VisibilityMPH" REAL, \
                              "Wind Direction" varchar(15), \
                              "Wind SpeedMPH" REAL, \
                              "Gust SpeedMPH" REAL, \
                              "PrecipitationIn" REAL, \
                              "Events" varchar(20), \
                              "Conditions" varchar(20), \
                              "WindDirDegrees" REAL);'
                             )
        # Initialize DB
        _initialize_table(conn, tablename)
        # Remove duplicates from dataframe
        df = self.remove_dups(df=dataframe, tablename=tablename, 
                              engine=conn, dup_cols=['Time'])
        # Write it to the database
        self.logger.info("Writing to {new} entries to database".format(new=df.shape[0]))
        df.replace(['-', '\s+'], np.nan, regex=True)
        df.set_index(["Time"], inplace=True)
        df.to_sql(tablename, engine, if_exists='append')


if __name__ == "__main__":
    weather = Weather()
    pwsid   = "KNYNEWYO116"
    airport = "KNYC"
    dbname  = os.getenv("CUIP_WEATHER_DBNAME")
    # set start and end date ranges
    st_date = "2016.01.20"
    en_date = "2016.01.24"
    # create empty dataframes for storing results
    station_data = airport_data = weather_with_visibility = pd.DataFrame()
    # setting up pool of workers
    pool    = multiprocessing.Pool(2)
    # settin up arguments for workers
    start_date   = datetime.datetime(*map(int, st_date.split(".")))
    end_date     = datetime.datetime(*map(int, en_date.split(".")))
    numdays      = (end_date - start_date).days
    date_range   = [end_date - datetime.timedelta(days=x) for x in range(0, numdays)]
    # get weather info asynchronously
    for date in date_range:
        airport_data = airport_data.append(weather.from_airport(date.year, date.month, date.day, "KNYC"))
        # station_data = station_data.append(weather.from_weather_station(date.year, date.month, date.day, "KNYNEWYO116"))
    weather.to_database(dbname, "weather", airport_data)
    """
    # sorting data
    airport_data.sort(["Time"], inplace=True)
    station_data.sort(["Time"], inplace=True)
    # combining airport data with station for visibility field
    weather_with_visibility = pd.merge_asof(station_data,
                                            airport_data[['Time', 'VisibilityMPH']], 
                                            left_on='Time', right_on='Time', 
                                            tolerance=pd.Timedelta('59minutes'))
    """
