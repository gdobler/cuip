import psycopg2
import os
import time
import multiprocessing
from datetime import datetime, timedelta
from cuip.cuip.utils.misc import get_files
from cuip.cuip.utils import cuiplogger

# Logger
logger = cuiplogger.cuipLogger(loggername="DATABASE", tofile=False)
INPATH = '/projects/cusp/10101/0/'


class Worker(multiprocessing.Process):

    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.conn = psycopg2.connect("dbname='uosdr'")
        cur = self.conn.cursor()
        self.initialize_table()
        
    def initialize_table(self):
        """
        Create table schema and commit
        ..note: If the table exists, it will not perform any action
        """
        cur = self.conn.cursor()
        cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('lightscape',))
        if not cur.fetchone()[0]:
            # Table does not exist
            logger.warning("Creating New Table")
            cur.execute('''
CREATE TABLE lightscape
(id SERIAL,
gid INT NOT NULL, 
fname varchar NOT NULL, 
fpath varchar NOT NULL, 
mean REAL NOT NULL, 
std REAL NOT NULL, 
bright INT NOT NULL, 
time timestamp with time zone PRIMARY KEY NOT NULL);
''')
            self.conn.commit()

    def reset_table(self):
        """
        Delete all the contents of table and
        the table itself
        """
        cur = self.conn.cursor()
        cur.execute("DELETE FROM LIGHTSCAPE")
        cur.execute("DROP TABLE LIGHTSCAPE")
        conn.commit()

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                logger.info("%s: Exiting"%(proc_name))
                self.task_queue.task_done()
                break
            answer = next_task(connection=self.conn)
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


class AddEntry(object):

    def __init__(self, fgen):
        self.fgen = fgen

    def __call__(self, connection=None):
       """
       Add entry to the database
       """
       conn = connection
       cur = conn.cursor()
       try:
          for f in self.fgen:
              logger.info("Adding %s to Database"%(str(f)))
              proc_query = "INSERT INTO lightscape (gid, fname, fpath, mean, std, bright, time) VALUES (0, %s, %s, 0, 0, 0, %s);"
              cur.execute(proc_query, (os.path.basename(f), os.path.dirname(f), datetime.fromtimestamp(os.path.getmtime(f))))
       except psycopg2.IntegrityError:
           # Found Duplicate Entry.. Do Nothing
           logger.warning(str(f)+" already exists")
           conn.rollback()
       finally:
           conn.commit()
       

#fgen = getFiles(INPATH, s_date.strftime("%Y.%m.%d"), s_date.strftime("%H.%M.%S"), 
#                e_date.strftime("%Y.%m.%d"), e_date.strftime("%H.%M.%S"))

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
               get_files(INPATH,
                         start_datetime.strftime("%Y.%m.%d"), start_datetime.strftime("%H.%M.%S"),
                         start_next.strftime("%Y.%m.%d"), start_next.strftime("%H.%M.%S"))
               )
           start_datetime += timedelta(hours=incr)
       else:
           # range is between current time iteration and end
           files_gen_list.append(
               get_files(INPATH,
                         start_datetime.strftime("%Y.%m.%d"), start_datetime.strftime("%H.%M.%S"),
                         end_datetime.strftime("%Y.%m.%d"), end_datetime.strftime("%H.%M.%S"))
               )
           break
   return files_gen_list

if __name__ == "__main__":
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
  
    # Start consumers
    num_workers = 1
    logger.info('Creating %d Workers' % num_workers)
    workers = [ Worker(tasks, results)
                for i in xrange(num_workers) ]
    for w in workers:
        w.start()

    def poisonChildren():
        # Add a poison pill for each consumer
        for i in xrange(num_workers):
            tasks.put(None)

    # Enqueue job
    start_date = "2013.11.16"
    start_time = "23.59.59"
    end_date   = "2013.11.17"
    end_time   = "23.59.59"
    s_datetime  = datetime(*map(int, start_date.split('.') + start_time.split('.') ))
    e_datetime  = datetime(*map(int, end_date.split('.') + end_time.split('.') ))

    files_gen_list = get_file_generators(INPATH, s_datetime, e_datetime)
    #num_jobs = len(files_gen_list)

    try:
        for gens in files_gen_list:
            filelist = [f for f in gens]
            tasks.put(AddEntry(filelist))

    finally:
        poisonChildren()
        tasks.join()
