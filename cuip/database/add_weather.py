import os
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, exc
from sqlalchemy.sql import update
from cuip.cuip.utils import cuiplogger

# logger
logger = cuiplogger.cuipLogger(loggername="AddWeather", tofile=False)

class AddWeather(object):

    def __init__(self, w_dbname, w_tbname,
                 start_datetime, end_datetime):
        """
        Update database with the contents from Dataframe
        Parameters
        ----------
        df: pd.DatFrame()
            datframe to update the columns from
        .. note: dataframe should have `Time` column
            of datatype datetime
        """
        self.start_datetime = start_datetime
        self.end_datetime   = end_datetime
        self.wdf            = self.get_weather_df(w_dbname,
                                                  w_tbname)

    def __call__(self, session=None, engine=None, table=None):

        # load the data chunk from files database for similar time
        query = """SELECT * FROM test \
                 WHERE "timestamp" \
                 BETWEEN '{first}' AND '{last}'""".format(
                                                        first=self.start_datetime,
                                                        last=self.end_datetime)
        dbf   = pd.read_sql(query, engine)

        # sort the files database
        dbf.sort_values(by='timestamp', inplace=True)
        # sort the weather dataframe
        self.wdf.sort_values(by='Time', inplace=True)

        # combine the weather with the chunk
        combined_df = pd.merge_asof(dbf,
                                    self.wdf[['Time', 'VisibilityMPH', 'Conditions']],
                                    left_on='timestamp', right_on='Time',)
                                    #tolerance=pd.Timedelta('59minutes'))
        self.update_database(combined_df, session, table)

    def get_weather_df(self, w_dbname, w_tbname):
        w_engine = create_engine('postgresql:///{0}'.format(w_dbname))
        w_md     = MetaData(bind=w_engine)
        w_md.reflect()
        w_table  = Table(w_tbname, w_md, autoload=True)
        query = """SELECT * FROM {table} \
                 WHERE "Time" \
                 BETWEEN '{first}' AND '{last}'""".format(table=w_tbname,
                                                        first=self.start_datetime,
                                                        last=self.end_datetime)
        wdf      = pd.read_sql(query, w_engine)
        return wdf

    def update_database(self, df, session, table):
        """
        Parameters
        ----------
        df: pd.DataFrame()
            dataframe to `Upsert` columns
        session: sqlalchemy session
        table: table in which to update
        """
        # update the database
        combined_df = df
        try:
            for ind, row in combined_df.iterrows():
                upd = update(table) \
                    .values({'visibility': row['VisibilityMPH'],
                             'conditions'   : row['Conditions']}) \
                             .where(table.c.timestamp == row['timestamp'])
                session.execute(upd)
                session.commit()
        except exc.IntegrityError:
            logger.warning("Duplicate Entry found for "+str(os.path.basename(f)))
            session.rollback()
