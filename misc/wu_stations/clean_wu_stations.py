#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Clean the Weather Underground station codes file that was grabbed
# from: https://www.wunderground.com/us/ny/new-york/zmw:10002.1.99999
# The produced output is a pipe delimited file

if __name__=="__main__":

    # -- get lines from file (only those with NY in the station code
    lines = [line for line in open("wu_station_codes_raw.txt") if "NY" in line]

    # -- split to pull of station code
    lsplit = [line.split(")") for line in lines]

    # -- set the output lines (including header)
    out = ["name|code\n"]+["|".join(line[0].split("(")).replace(" |","|")+
                           "\n" for line in lsplit]

    # -- write to file
    with open("wu_ny_stations.pdt","w") as fopen:
        dum = [fopen.write(line) for line in out]
