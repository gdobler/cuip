import os
import multiprocessing
import datetime
from sqlalchemy import create_engine
from cuip.cuip.utils import cuiplogger
from cuip.cuip.database import db_tables
from cuip.cuip.utils.misc import get_file_generators
from cuip.cuip.database.add_files import AddTask
from cuip.cuip.database.add_weather import AddWeather
from cuip.cuip.database.database_worker import Worker

# logger
logger = cuiplogger.cuipLogger(loggername="CreateDB", tofile=False)

if __name__ == "__main__":
    inpath    = os.getenv("CUIP_2013")
    dbname    = os.getenv("CUIP_DBNAME")
    w_dbname  = os.getenv("CUIP_WEATHER_DBNAME")
    tablename = "test"
    w_tbname  = "knyc"

    # sanity checks for database tables
    logger.info("Checking if table schema exists")
    engine = create_engine('postgresql:///{0}'.format(dbname))
    uo_files_table = db_tables.ToFilesDB
    if not uo_files_table.__table__.exists(bind=engine):
        uo_files_table.__table__.create(bind=engine, checkfirst=False)
    weather_table  = db_tables.ToWeatherDB
    if not weather_table.__table__.exists(bind=engine):
        weather_table.__table__.create(bind=engine, checkfirst=False)

    # threadsafe queues for adding tasks and collecting results
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    # start database workers
    num_workers = 16
    logger.info('Creating %d Workers' % num_workers)
    workers = [ Worker(dbname, tablename, tasks, results)
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
    end_date    = "2013.11.30"
    end_time    = "00.09.59"
    s_datetime  = datetime.datetime(*map(int, start_date.split('.') + 
                                         start_time.split('.') ))
    e_datetime  = datetime.datetime(*map(int, end_date.split('.') + 
                                         end_time.split('.') ))

    # for adding files to the database
    files_gen_list = get_file_generators(inpath, s_datetime, e_datetime)

    # for adding weather, split dates into [start, end] range
    numdays      = (e_datetime - s_datetime).days
    dates        = [e_datetime - datetime.timedelta(days=x) \
                        for x in range(0, numdays)]
    if numdays == 1:
        date_range = [[e_datetime, s_datetime]]
    else:
        date_range = zip(*(iter(dates), )* 2)

    try:
        # adding weather info from weather database
        #for date in date_range:
        #    tasks.put(AddWeather(w_dbname, w_tbname,
        #                         date[1], date[0]))
        # adding new file entry
        for gens in files_gen_list:
            filelist = [f for f in gens]
            tasks.put(AddTask(filelist))
    finally:
        poisonChildren()
        tasks.join()
