import os
import psycopg2
from datetime import datetime, timedelta
from cuip.cuip.utils import cuiplogger
logger = cuiplogger.cuipLogger(loggername="MISC", tofile=False)


def get_files(dbname, start_datetime, end_datetime):
    """
    Fetch all files between `start_datetime`, and
    `end_datetime` from database `dbname`
    .. note: If database of filenames does not exist,
             use _get_files method.
    Parameters
    ----------
    dbname: str
        postgres database name
    start_datetime: `datetime.datetime`
        start datetime from which to get the files
    end_datetime: `datetime.datetime`
        end datetime until which to get the files
    Returns
    -------
    files: list
        list of absolute paths of the files
    """
    conn   = psycopg2.connect("dbname='%s'"%(dbname))
    cur    = conn.cursor()
    querry = "SELECT fname, fpath \
              FROM lightscape     \
              WHERE timestamp     \
              BETWEEN %(start)s and %(end)s;"
    cur.execute(querry, {'start': start_datetime, 
                         'end'  : end_datetime})
    rows   = cur.fetchall()
    # join filename with filepath
    files  = map(lambda x: os.path.join(x[1], x[0]), rows)
    return files

def _pathrange(basepath, start, end, delta):
    """
    Generate paths for datetime between `start` and `end`
    in the `YYYY/MM/DD/HH.MM.SS` format
    .. note: Use this only if database of files does not exist
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
            # Only proceed if path with year/month/day exists
            if os.path.exists("{path}/{year}/{month:0>2}/{day:0>2}".format(path = basepath, 
                                                                           year = curr.year, 
                                                                           month = curr.month, 
                                                                           day = curr.day)):
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
            else:
                curr += timedelta(days=1)
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

def _get_files(path, start_date, start_time, end_date, end_time):
    """
    Fetch all files between `start_date`, `start_time` and
    `end_date`, `end_time`
    .. note: Use this only if database of files does not exist.
             If database exists, use get_files method
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
    st  = datetime(*[int(i) for i in start_date.split(".") + start_time.split(".")])
    end = datetime(*[int(i) for i in end_date.split(".") + end_time.split(".")])

    paths = pathrange(os.path.abspath(path), 
                      st, end, timedelta(seconds=1))

    try:
        while True:
            next_path = paths.next()
            for files in os.listdir(next_path):
                yield os.path.join(next_path, files)
    except StopIteration as si:
        pass
    except OSError as ose:
        logger.error("Path "+ next_path +" doesn't exist")
