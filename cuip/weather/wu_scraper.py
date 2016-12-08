#!/usr / bin / env python
# -*- coding: utf-8 -*-
"""
This module is responsible for scraping the Weather Underground
for selected pws. The module will grab the output and store it 
in the csv format. To get help on running this module,

python wu_scraper.py -h

"""

from __future__ import print_function
import argparse
import os
import sys
import pandas as pd
import datetime
from cuip import __version__
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

    def get_weather(self, year, month, day, pwsid="KNYNEWYO116", airport="kjfk"):
        """
        Grab the weather data from Weather Underground
        
        Parameters
        ----------
        year : int
        month : int
        day : int
        pwsid : str
        weather underground's personal weather station id
        the default value is "KNYNEWYO116" for BoreumHill NYC
        airport: str
        airport code prefixed by 'k'. default is 'kjfk'. This 
        is used for obtaining visibility
        Returns
        -------
        Pandas DataFrame containing weather data for particular day.
        """
        # Create urls
        station_url  = self.station_url.format(pwsid=pwsid, year=year, month=month, day=day)
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
        # fetch weather data
        station_cols = pd.read_csv(station_url, nrows=1).columns                                                                                                   
        station_data = pd.read_csv(station_url, names=station_cols[:-1], 
                                   usecols=station_cols[:-1], parse_dates=["Time"], 
                                   infer_datetime_format=True)
        airport_cols = pd.read_csv(airport_url, nrows=1).columns
        time_col_name = set(air_time_cols).intersection(airport_cols).pop()
        airport_data = pd.read_csv(airport_url, names=airport_cols[:-1], 
                                   usecols=airport_cols[:-1], parse_dates=[time_col_name], 
                                   infer_datetime_format=False, date_parser=airport_dateparser, 
                                   skiprows=2)

        # convert string timestamp to datetime
        station_data = station_data[1::2]
        station_data['Time'] = pd.to_datetime(station_data['Time'])
        airport_data.rename(columns={time_col_name: 'Time'}, inplace=True)
        
        # merge weather station data with airport weather data
        weather_with_visibility = pd.merge_asof(station_data, 
                                                airport_data[['Time', 'VisibilityMPH']], 
                                                left_on='Time', right_on='Time', 
                                                tolerance=pd.Timedelta('59minutes'))
        return weather_with_visibility

    def write_weather(self, data, fname):
        """
        Write the weather data to a database.
        This will currently create a csv file. 
        In the final implementations, it will push data
        to csv file.
        
        Paramters
        ---------
        data : pd.DataFrame
        fname: str
        Name and path of the file to write the data
        
        Returns
        -------
        None
        """

        # place-holder to write the weather data to a database
        data.to_csv(fname, encoding='utf-8')
        self.logger.info("Written weather data to: " + str(fname))


    def weather_today(self, pwsid="KNYNEWYO116", airport="kjfk"):
        """
        Print current weather information
        
        Parameters
        ----------
        pwsid : str
        weather underground's personal weather station id
        the default value is "KNYNEWYO116" for BoreumHill NYC
        airport: str
        airport code prefixed by 'k'. default is 'kjfk'. This
        is used for obtaining visibility
        
        Returns
        -------
        None
        """
        now = datetime.datetime.now()
        current_weather = self.get_weather(year=now.year, month=now.month, 
                                           day=now.day, pwsid=pwsid, 
                                           airport=airport)
        for i in current_weather.columns:
            print("{col: <25} {info}".format(col=i, info=current_weather[i][-1]))

def _parse_args(args):
    """
    Parse command-line args

    Paramters
    ---------
    args : list 
        Commandline paramters as list of strings

    Returns
    -------
    Commandline paramters as :obj:`airgparse.Namespace`
    """
    parser = argparse.ArgumentParser(
        description="Grab the Weatherdata from WeatherUnderground")
    parser.add_argument(
        "-v",
        "--version",
        action='version',
        version='cuip {ver}'.format(ver=__version__))
    parser.add_argument(
        "-t",
        "--today",
        action='store_true',
        dest='today',
        help="""Get current weather information. Therer is a bug with 
Weather underground's reporting system so don't use it between midnight and 2 am.""")
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        dest='write_switch',
        help="Write the weather info to a csv file"
    )
    parser.add_argument(
        "-n",
        "--name",
        action='store',
        dest='csvname',
        default='.',
        type=str,
        help="full path for csv file (default = '.')"
    )
    parser.add_argument(
        "-y",
        "--year",
        action='store',
        dest='year',
        type=int,
        help="Year"
    )
    parser.add_argument(
        "-m",
        "--month",
        action='store',
        dest='month',
        type=int,
        help="Month"
    )
    parser.add_argument(
        "-d",
        "--day",
        action='store',
        dest='day',
        type=int,
        help="day"
    )
    parser.add_argument(
        "-p",
        "--pwsid",
        action='store',
        dest='pwsid',
        default="KNYNEWYO116",
        type=str,
        help="personal weather station id"
    )
    parser.add_argument(
        "-a",
        "--air",
        action='store',
        dest='airport',
        default="KJFK",
        type=str,
        help="airport id prefixed with k. eg. kjfk"
    )

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    print("Starting wu_scraper")
    weather_obj = Weather()
    arg         = _parse_args(args)
    
    if arg.today:
        weather_obj.weather_today(arg.pwsid, arg.airport)
    elif arg.write_switch:
        weather = weather_obj.get_weather(year    = arg.year,
                                          month   = arg.month,
                                          day     = arg.day,
                                          pwsid   = arg.pwsid,
                                          airport = arg.airport)
        weather_obj.write_weather(w, arg.csvname)
    else:
        with pd.option_context('display.max_rows', 999, 'display.max_columns', 7):
            print(weather_obj.get_weather(year    = arg.year,
                                          month   = arg.month,
                                          day     = arg.day,
                                          pwsid   = arg.pwsid,
                                          airport = arg.airport))

def run():
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
