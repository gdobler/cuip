#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

if __name__ == "__main__":

    for ii in range(10):
        cmdl = "mv output/register_{0:04}.log output/register_{0:04}_3.log"
        cmdc = "mv output/register_{0:04}.csv output/register_{0:04}_3.csv"
        os.system(cmdl.format(ii))
        os.system(cmdc.format(ii))
