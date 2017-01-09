import os
from datetime import datetime
from sqlalchemy import exc, update
from cuip.cuip.utils import cuiplogger
from cuip.cuip.database.db_tables import ToFilesDB

# logger
logger = cuiplogger.cuipLogger(loggername="AddFiles", tofile=False)

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
            update(ToFilesDB).where(ToFilesDB.fname.in_(map(lambda x: x.split('/')[-1], self.flist))).values(self.values_to_update)
            session.commit()
        except exc.IntegrityError:
            logger.warning("Duplicate Entry found for "+str(os.path.basename(f)))
            session.rollback()
