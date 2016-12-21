import os
from datetime import datetime
from sqlalchemy import exc
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
                                timestamp=datetime.fromtimestamp(
                        os.path.getmtime(f)).replace( microsecond=0),
                                cloud='',
                                visibility=-1,
                                roffset=0, coffset=0, angle=0,
                                usable=False)
                session.add(row)
                session.commit()
            except exc.IntegrityError:
                logger.warning("Duplicate Entry found for "+str(os.path.basename(f)))
                session.rollback()
