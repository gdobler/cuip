import os
from datetime import datetime, timedelta
from cuip.cuip.utils import cuiplogger
logger = cuiplogger.cuipLogger(loggername="MISC", tofile=False)

def pathrange(basepath, start, end, delta):
    """
    Generate paths for datetime between `start` and `end`
    in the `YYYY/MM/DD/HH.MM.SS` format
    Parameters
    ----------
    basepath: str
        basepath
    start: `datetime.datetime`
    end: `datetime.datetime`
    delta: `datetime.timedelta`
    
    Returns
    -------
    `Generator` containing paths
    """
    curr = start
    try:
        while curr < end:
            next_path = "{path}/{year}/{month:0>2}/{day:0>2}/{hour:0>2}.{minute:0>2}.{second:0>2}".format(path = basepath,
                                                                                                          year = curr.year,
                                                                                                          month = curr.month,
                                                                                                          day = curr.day,
                                                                                                          hour = curr.hour,
                                                                                                          minute = curr.minute,
                                                                                                          second = curr.second)
            if os.path.exists(next_path):
                yield next_path
                
            curr += delta
    except Exception as ex:
        logger.error("Error in path_range: "+str(ex))

        # os.path.join doesn't respect the formatting
        # Leaving the code here just to remember
        """
        yield os.path.join(str(basepath), 
                           str(curr.year), 
                           str(curr.month), 
                           str(curr.day), 
                           "{hour:0>2}.{minute:0>2}.{second:0>2}".format(
                hour=curr.hour, 
                minute=curr.minute, 
                second=curr.second
                ))
        curr += delta
        """

def get_files(path, start_date, start_time, end_date, end_time):
    """
    Fetch all files between `start_date`, `start_time` and
    `end_date`, `end_time`
    Parameters
    ----------
    path: str
        root location where the files are stored.
        .. note: This should be the root dir from where the
                 directory structure looks like 
                 `YYYY/MM/DD/HH.MM.SS/file`
    start_date: str
    end_date: str
        format: YYYY.MM.DD
    start_time: str
    end_time: str
        format: HH.MM.SS
    """
    s_year, s_month, s_day = start_date.split('.')
    s_hour, s_min, s_sec = start_time.split('.')
    e_year, e_month, e_day = end_date.split('.')
    e_hour, e_min, e_sec = end_time.split('.')

    paths = pathrange(os.path.abspath(path), 
                      datetime(int(s_year), int(s_month), int(s_day), int(s_hour), int(s_min), int(s_sec)), 
                      datetime(int(e_year), int(e_month), int(e_day), int(e_hour), int(e_min), int(e_sec)), 
                      timedelta(seconds=1)
                      )

    try:
        while True:
            next_path = paths.next()
            for files in os.listdir(next_path):
                yield os.path.join(next_path, files)
    except StopIteration as si:
        pass
    except OSError as ose:
        logger.error("Path "+ next_path +" doesn't exist")
