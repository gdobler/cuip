import os
import csv
from datetime import datetime
from sqlalchemy import exc, update
from cuip.cuip.utils import cuiplogger
from cuip.cuip.database.db_tables import ToFilesDB

# logger
logger = cuiplogger.cuipLogger(loggername="AddFiles", tofile=False)
outpath = 'output/combined_images'
class AddTask(object):

    def __init__(self, fgen):
        self.fgen = fgen

    def __call__(self, session=None, *args, **kwargs):
        """
        Add database task
        """
        for f in self.fgen:
            try:
                row = ToFilesDB(gid=9,
                                fname=os.path.basename(f),
                                fpath=os.path.dirname(f),
                                fsize=int(os.path.getsize(f)/1024**2.),
                                mean=0, std=0, bright_pix=0,
                                timestamp=datetime.fromtimestamp(os.path.getmtime(f)),
                                cloud='',
                                visibility=-1,
                                roffset=0, coffset=0, angle=0,
                                usable=False)
                session.add(row)
                session.commit()
            except exc.IntegrityError:
                logger.warning("Duplicate Entry found for "+str(os.path.basename(f)))
                session.rollback()

class UpdateTask(object):
    
    def __init__(self, flist, values_to_update):
        """
        Parameters
        ----------
        flist: list
            list of filenames
        values_to_update: dict
            key value pair of column/ parameter to update in database table
        """
        self.flist = flist
        self.values_to_update = values_to_update

    def __call__(self, session=None, *args, **kwargs):
        """
        update database
        """
        try:
            if self.flist:
                upd = update(ToFilesDB). \
                    where(ToFilesDB.fname.in_([os.path.basename(x) \
                                                   for x in self.flist])). \
                                                   values(self.values_to_update)
            else:
                logger.info("Updating "+str(self.values_to_update)+ "for all rows")
                upd = update(ToFilesDB).values(self.values_to_update)
            session.execute(upd)
            session.commit()
        except exc.IntegrityError:
            logger.warning("Integrity Error:"+ 
                           "Possibly duplicate entry found."+
                           "Rolling back the commit before exiting")
            session.rollback()

class ToCSV(object):
    
    def __init__(self, where_clause=None, compare_value=None):
        """
        Parameters
        ----------
        where_clause: parameter/ column of the table
        compare_value: list
            list of values for comparing `where_clause`
        """
        logger.info("Checking attribute ")
        self.where = where_clause
        self.compare_value = compare_value

    def __call__(self, session=None, *args, **kwargs):
        logger.info("writing to csv")
        query = session.query(ToFilesDB). \
            filter(getattr(ToFilesDB, self.where, 'gid'). \
                       in_(self.compare_value))
        exc = session.execute(query)
        result = exc.fetchall()
        fh = open(os.path.join(outpath, '{0}.csv'. \
                                   format(datetime.now(). \
                                              isoformat())), 'wb')
        outcsv = csv.writer(fh)
        outcsv.writerow(exc.keys())
        outcsv.writerows(result)
        fh.close()
