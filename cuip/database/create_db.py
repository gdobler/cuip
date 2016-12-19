import psycopg2
import os
import time
import multiprocessing
from psycopg2.extensions import AsIs
from datetime import datetime, timedelta
from cuip.cuip.utils.misc import _get_files
from cuip.cuip.utils import cuiplogger

# Logger
logger = cuiplogger.cuipLogger(loggername="DATABASE", tofile=False)


class Worker(multiprocessing.Process):

    def __init__(self, dbname, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.conn = psycopg2.connect("dbname='%s'"%(dbname))
        cur = self.conn.cursor()
        self.initialize_table()
        
    def initialize_table(self):
        """
        Create table schema and commit
        ..note: If the table exists, it will not perform any action
        """
        cur = self.conn.cursor()
        cur.execute("select exists(SELECT * FROM information_schema.tables \
                    WHERE table_name=%s)", ('uo_files',))
        if not cur.fetchone()[0]:
            # Table does not exist
            logger.warning("Creating New Table")
            cur.execute(" \
                       CREATE TABLE  uo_files\
                       (id SERIAL, \
                       gid INT NOT NULL, \
                       fname varchar NOT NULL, \
                       fpath varchar NOT NULL, \
                       mean REAL NOT NULL, \
                       std REAL NOT NULL, \
                       bright_pix INT NOT NULL, \
                       timestamp timestamp without time zone PRIMARY KEY NOT NULL, \
                       visibility REAL NOT NULL, \
                       weather varchar NOT NULL, \
                       xoffset INT NOT NULL, \
                       yoffset INT NOT NULL, \
                       angle REAL NOT NULL); \
                       ")
            self.conn.commit()

    def reset_table(self):
        """
        Delete all the contents of table and
        the table itself
        """
        cur = self.conn.cursor()
        cur.execute("DELETE FROM uo_files")
        cur.execute("DROP TABLE uo_files")
        conn.commit()

    def get_columns(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM uo_files LIMIT 0")
        colnames = [desc[0] for desc in cur.description]
        return colnames

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                logger.info("%s: Exiting"%(proc_name))
                self.task_queue.task_done()
                break
            answer = next_task(connection=self.conn, 
                               colnames=self.get_columns)
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class AddFile(object):

    def __init__(self, fgen):
        self.fgen = fgen

    def __call__(self, connection=None, colnames=None):
       """
       Add entry to the database
       """
       conn = connection
       cur = conn.cursor()
       for f in self.fgen:
           try:
               logger.info("Adding %s to Database"%(str(f)))
               proc_query = "INSERT INTO uo_files ( \
                             gid, fname, fpath, mean, \
                             std, bright_pix, timestamp, \
                             visibility, weather, xoffset, \
                             yoffset, angle) \
                             VALUES (0, %s, %s, 0, \
                             0, 0, %s, \
                             0, %s, 0, \
                             0, 0);"
               cur.execute(proc_query, (os.path.basename(f), 
                                        os.path.dirname(f), 
                                        datetime.fromtimestamp(os.path.getmtime(f)).\
                                            replace(microsecond=0), "NA"))
           except psycopg2.IntegrityError:
               # Found Duplicate Entry.. Do Nothing
               logger.warning(str(f)+" already exists")
               conn.rollback()
       # Commit all the changes
       conn.commit()


class UpdateFileEntry(object):
    
    def __init__(self, timestamp=None, 
                 filename=None, filepath=None, **kwargs):
        """
        Parameters
        ----------
        timestamp: `datetime.datetime`, optional
        filename: str
        filepath: str
            absolute path of the file
        **kwargs: key value pair of fields to
            be updated
        .. note: Either timestamp of filename & filepath must
                 be provided
        """
        self.timestamp = timestamp.replace(microsecond=0)
        self.fname     = filename
        self.fpath     = filepath
        self.kwargs    = kwargs

    def __call__(self, connection=None, colnames=None):
        """
        Update entry in the database
        """
        conn = connection
        cur = conn.cursor()
        try:
            for val in self.kwargs.items():
                proc_query = "UPDATE uo_files SET %s=%s"
                if self.timestamp:
                    proc_query += " WHERE timestamp=%s;"
                    cur.execute(proc_query, (AsIs(val[0]), val[1],
                                             self.timestamp))
                elif self.fname and self.fpath:
                    proc_query += " WHERE fname=%s AND fpath=%s;"
                    cur.execute(proc_query, (AsIs(val[0]), str(val[1]), 
                                             self.fname, self.fpath))
        except Exception as ex:
            logger.warning("Error updating database: "+str(ex))
            conn.rollback()
        finally:
            conn.commit()


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

if __name__ == "__main__":
    inpath = os.getenv("CUIP_2013")
    dbname = os.getenv("CUIP_DBNAME")

    # threadsafe queues for adding tasks and collecting results
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
  
    # start workers
    num_workers = 1
    logger.info('Creating %d Workers' % num_workers)
    workers = [ Worker(dbname, tasks, results)
                for i in xrange(num_workers) ]
    for w in workers:
        w.start()

    def poisonChildren():
        # Add a poison pill for each consumer
        for i in xrange(num_workers):
            tasks.put(None)

    # Enqueue job
    start_date  = "2013.11.17"
    start_time  = "00.00.01"
    end_date    = "2013.11.17"
    end_time    = "00.09.59"
    s_datetime  = datetime(*map(int, start_date.split('.') + start_time.split('.') ))
    e_datetime  = datetime(*map(int, end_date.split('.') + end_time.split('.') ))

    files_gen_list = get_file_generators(inpath, s_datetime, e_datetime)

    try:
        # adding new file entry
        for gens in files_gen_list:
            filelist = [f for f in gens]
            tasks.put(AddFile(filelist))
            #tasks.put(UpdateFileEntry(timestamp=datetime(*map(int, "2013.11.17".split('.') + "00.09.10".split('.'))), 
            #                          filename=None, filepath=None, visibility=9))
        # updating weather entry
    finally:
        poisonChildren()
        tasks.join()
