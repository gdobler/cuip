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

logger = cuiplogger.cuipLogger(loggername="wu_scraper", tofile=False)
URL_TEMPLATE = "http://www.wunderground.com/weatherstation/WXDailyHistory.asp?ID={pwsid}&day={day}&month={month}&year={year}&graphspan=day&format=1"


def get_weather(year, month, day, pwsid="KNYNEWYO116"):
    """
    Grab the weather data from Weather Underground

    Parameters
    ----------
    year : int
    month : int
    day : int
    pwsid : str 
        the default value is "KNYNEWYO116" for BoreumHill NYC

    Returns
    ------
    Pandas DataFrame containing weather data for particular day.
    """

    url = URL_TEMPLATE.format(pwsid=pwsid, year=year, month=month, day=day)
    cols = pd.read_csv(url, nrows=1).columns
    data = pd.read_csv(url, names=cols[:-1], header=False, skiprows=1, usecols=cols[:-1])
    return data[::2].set_index(['Time'])


def write_weather(data, fname):
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

    # Place-holder to write the weather data to a database
    data.to_csv(fname, encoding='utf-8')
    logger.info("Written weather data to: " + str(fname))


def weather_today(pwsid="KNYNEWYO116"):
    """
    Print current weather information

    Parameters
    ----------
    pwsid : str 
        the default value is "KNYNEWYO116" for BoreumHill NYC

    Returns
    -------
    None
    """
    now = datetime.datetime.now()
    current_weather = get_weather(year=now.year, month=now.month, day=now.day, pwsid=pwsid)
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
        help="Get current weather information")
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
    if len(args) == 0:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger.info("Starting wu_scraper")
    arg = _parse_args(args)
    if arg.today:
        weather_today(arg.pwsid)
    elif arg.write_switch:
        w = get_weather(year=arg.year,
                        month=arg.month,
                        day=arg.day,
                        pwsid=arg.pwsid)
        write_weather(w, arg.csvname)


def run():
    main(sys.argv[1:])

if __name__ == "__main__":
    run()
