import os
import psycopg2
import multiprocessing
import pandas as pd
from datetime import datetime, timedelta
from cuip.cuip.utils import cuiplogger
logger = cuiplogger.cuipLogger(loggername="MISC", tofile=False)


class asynchronous(object):
    """
    simple decorator for applying multiprocessing
    on functions
    """
    def __init__(self, func):
        self.func = func

        def processified(*args, **kwargs):
            self.tasks.put(self.func(*args, **kwargs))

        self.processified = processified

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def start(self, *args, **kwargs):
        self.tasks = multiprocessing.Queue()
        proc = multiprocessing.Process(target=self.processified, args=args, kwargs=kwargs)
        proc.start()
        return asynchronous.Result(self.tasks, proc)


    class NotYetDoneException(Exception):
        def __init__(self, message):
            self.message = message


    class Result(object):
        def __init__(self, tasks, process):
            self.tasks = tasks
            self.process = process

        def is_done(self):
            return not self.process.is_alive()

        def get_result(self):
            if not self.is_done():
                raise asynchronous.NotYetDoneException("call has not yet completed the task")

            if not hasattr(self, 'result'):
                self.result = self.tasks.get()

            else:
                self.process.join()
                self.process.close()

            return self.result


def get_files(dbname, start_datetime, end_datetime, df=False):
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
    f_tbname = os.getenv("CUIP_TBNAME")
    conn     = psycopg2.connect("dbname='%s'"%(dbname))
    cur      = conn.cursor()
    query    = "SELECT fname, fpath, fnumber, timestamp \
                FROM {tbname}     \
                WHERE timestamp     \
                BETWEEN %(start)s and %(end)s ORDER BY timestamp;" \
        .format(tbname=f_tbname)

    if not df:
        cur.execute(query, {'start': start_datetime, 'end': end_datetime})
        return [(os.path.join(x[1], x[0]), x[2]) for x in cur.fetchall()]
    else:
        return pd.read_sql_query(query, conn, params={'start': start_datetime, 
                                                      'end': end_datetime})

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

def _get_files(path, start_datetime, end_datetime):
    """
    Fetch all files between `start_datetime`, and
    `end_datetime`
    .. note: Use this only if database of files does not exist.
             If database exists, use get_files method
    Parameters
    ----------
    path: str
        root location where the files are stored.
        .. note: This should be the root dir from where the
                 directory structure looks like
                 `YYYY/MM/DD/HH.MM.SS/file`
    start_datetime: `datetime.datetime`
        start datetime from which to get the files
    end_datetime: `datetime.datetime`
        end datetime until which to get the files
    """
    paths = _pathrange(os.path.abspath(path),
                      start_datetime, end_datetime,
                      timedelta(seconds=1))
    try:
        while True:
            next_path = paths.next()
            for files in os.listdir(next_path):
                yield os.path.join(next_path, files)
    except StopIteration as si:
        pass
    except OSError as ose:
        logger.error("Path "+ next_path +" doesn't exist")

def get_file_generators(inpath, start_datetime, end_datetime, incr=1):
   """
   Returns generators containing files split by the hour
   between start and end datetime
   Parameters
   ----------
   inpath: str
       string containing path to the root directory where
       the files are stored
   start_datetime: `datetime.datetime`
       datetime object containing start period
   end_datetime: `datetime.datetime`
       datetime object containgin end period
   """
   files_gen_list = []
   while True:
       start_next = start_datetime+timedelta(hours=incr)
       if start_next <= end_datetime:
           # range is between current time interation and +incr hours
           files_gen_list.append(
               _get_files(inpath,
                         start_datetime,
                         start_next))
           start_datetime += timedelta(hours=incr)
       else:
           # range is between current time iteration and end
           files_gen_list.append(
               _get_files(inpath,
                         start_datetime,
                         end_datetime))
           break
   return files_gen_list


def query_db(dbname, start_datetime, end_datetime, columns=[]):
    """
    Grab columns from dbname.
    """
    f_tbname = os.getenv("CUIP_TBNAME")
    conn     = psycopg2.connect("dbname='%s'"%(dbname))
    cur      = conn.cursor()

    # -- if columns is empty, print columns
    if len(columns) == 0:
        query = "SELECT * from {tbname} LIMIT 0;".format(tbname=f_tbname)
        cur.execute(query)

        print("Columns available in {0} are:".format(f_tbname))
        for col in [i[0] for i in cur.description]:
            print(col)

        return

    cols  = ",".join(columns)
    query = "SELECT {colstr} \
             FROM {tbname}     \
             WHERE timestamp     \
             BETWEEN %(start)s and %(end)s ORDER BY timestamp;" \
        .format(colstr=cols, tbname=f_tbname)

    return pd.read_sql_query(query, conn, params={'start': start_datetime, 
                                                  'end': end_datetime})
