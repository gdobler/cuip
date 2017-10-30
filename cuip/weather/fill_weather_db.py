#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import bs4
import datetime
import pandas as pd
from sqlalchemy import create_engine


def get_wu_html():
    """ Get the Weather Underground HTML files. """

    # -- get date range
    st    = datetime.datetime(2013, 10, 1)
    en    = datetime.datetime(2017, 10, 29)
    nday  = (en - st).days + 1
    dlist = [d for d in (st + datetime.timedelta(i) for i in range(nday))]

    # -- grab html files
    opath = os.path.join("output", "wunderhtml")
    hbase = "https://www.wunderground.com/history/airport/KNYC/" \
            "{0:04}/{1:02}/{2:02}/DailyHistory.html"
    for date in dlist:
        yr    = date.year
        mo    = date.month
        dy    = date.day
        html  = hbase.format(yr, mo, dy)
        hfile = os.path.join(opath, "DailyHistory_{0:04}_{1:02}_{2:02}.html" \
                             .format(yr, mo, dy))

        if not os.path.isfile(hfile):
            os.system("wget {0}".format(html))
            os.system("mv DailyHistory.html {0}".format(hfile))

    return


def parse_daily_precipitation(soup):
    """ Pull off the daily precipitation total. """

    # -- find the table row
    attrs = {"id" : "historyTable"}
    prow  = [i.find_all("td") for i in soup.find("table", attrs=attrs) \
             .find("tbody").find_all("tr") if "Precipitation" in i.text][1]

    # -- get the daily value column text, encode ascii, strip, cast as float
    val = prow[1].text.encode("ascii", "ignore").replace("\n", "") \
                                            .replace("in", "").replace(" ", "")

    # -- per WU, "T" stands for trace precipitation detected
    if valt == "T":
        return 0.0
    else:
        return float(val)


def parse_wu_table(yr, mo, dy):
    """ Parse a Weather Underground HTML table. """

    # -- set the file
    html  = os.path.join("output", "wunderhtml",
                         "DailyHistory_{0:04}_{1:02}_{2:02}.html" \
                         .format(yr, mo, dy))
    fopen = open(html, "r")
    soup  = bs4.BeautifulSoup(fopen, "html.parser")

    # -- get header
    hdr = [i.text for i in soup.find("table",
                                    attrs={"class" : "obs-table responsive"}) \
           .find("thead").find_all("tr")[0].find_all("th")]

    # -- get the hourly weather table from html
    rows = soup.find("table", attrs={"class" : "obs-table responsive"}) \
               .find("tbody").find_all("tr")
    tbl  = [[ele.text.strip() for ele in row.find_all("td")] for row in rows]
    fopen.close()

    # -- convert to dataframe
    if any(["EDT" in i for i in hdr]):
        cols = ["Time (EDT)", "Temp.", "Humidity", "Precip"]
    else:
        cols = ["Time (EST)", "Temp.", "Humidity", "Precip"]
    data = pd.DataFrame(tbl, columns=hdr)[cols]
    data.columns = ["time", "temp", "humidity", "precip"]
    
    # -- parse columns
    def time_to_datetime(tstr):
        """ Convert Weather Underground EST to datetime. """

        return datetime.datetime.strptime("{0:04}/{1:02}/{2:02} " \
                                          .format(yr, mo, dy) + tstr,
                                          "%Y/%m/%d %I:%M %p")

    data["time"]     = data["time"].apply(time_to_datetime)
    data["temp"]     = pd.to_numeric(data["temp"] \
                                .apply(lambda x: x.encode("ascii", "ignore") \
                                        .replace("F", "")), errors="coerce")
    data["humidity"] = pd.to_numeric([i[:-1] for i in
                                      data["humidity"]], errors="coerce")
    data["precip"]   = [0.0 if i == "N/A" else float(i[:-3]) for i in
                        data["precip"]]

    # -- add daily precipitation
    data["daily_precip"] = [parse_daily_precipitation(soup)] * len(data)

    return data


def fill_weather_db():
    """ Fill the Weather Underground weather database. """

    # -- set the date range
    st    = datetime.datetime(2013, 10, 1)
    en    = datetime.datetime(2017, 10, 27)
    nday  = (en - st).days + 1
    dlist = [d for d in (st + datetime.timedelta(i) for i in range(nday))]

    # -- initialize the engine
    engine = create_engine("postgresql:///weather_underground")
    
    # -- add each day to database
    for ii, dd in enumerate(dlist):
        if (ii + 1) % 100 == 0:
            print("\radding day {0} of {1}".format(ii + 1, nday)),
            sys.stdout.flush()

        yr   = dd.year
        mo   = dd.month
        dy   = dd.day
        data = parse_wu_table(yr, mo, dy)

        data.to_sql("knyc_conditions", engine, if_exists="append", index=False)

    return


if __name__ == "__main__":

    # -- run it
    get_wu_html()
    fill_weather_db()
