import multiprocessing
from sqlalchemy import create_engine, MetaData, Table, exc
from sqlalchemy.orm import sessionmaker
from cuip.cuip.utils import cuiplogger

# logger
logger = cuiplogger.cuipLogger(loggername="DBWorker", tofile=False)

class Worker(multiprocessing.Process):

    def __init__(self, dbname, tablename, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.tablename = tablename
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.engine = create_engine('postgresql:///{0}'.format(dbname))
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self.md = MetaData(bind=self.engine)
        self.md.reflect()
        # get table
        self.table = self.initialize_table()

    def initialize_table(self):
        """
        Create table schema and commit
        ..note: If the table exists, it will not perform any action
        """
        try:
            table = Table(self.tablename, self.md, autoload=True)
            return table
        except exc.NoSuchTableError:
            logger.error(str(tablename)+" does not exist")

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                logger.info("%s: Exiting"%(proc_name))
                self.task_queue.task_done()
                break
            answer = next_task(session=self.session,
                               engine=self.engine,
                               table=self.table)
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return
