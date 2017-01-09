import os
import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, MetaData, Table, exc
from sqlalchemy.sql import update
from sqlalchemy.sql.expression import bindparam
from cuip.cuip.database.db_tables import ToFilesDB
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
        query = """SELECT * FROM uo_files \
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
        try:
            logger.info("Combining dataframes")
            if not (dbf.empty or self.wdf.empty):
            # -- for multiprocessing... TBD    
            #    dbf[['closest']] = dbf.timestamp.apply(
            #        self.find_closest_date, args=[self.wdf.Time])
            #    logger.info("merging")
            #    combined_df = pd.merge(dbf, self.wdf, left_on=['closest'], right_on=['Time'])
                combined_df = pd.merge_asof(dbf,
                                            self.wdf[['Time', 'VisibilityMPH', 'Conditions']],
                                            left_on='timestamp', right_on='Time',
                                            tolerance=pd.Timedelta('1hour'),
                                            )
                self.update_database(combined_df, session, table)
        except ValueError as ve:
            logger.warning("Error merging dataframe: "+str(ve))

    def find_closest_date(self, timepoint, time_series, add_time_delta_column=False):
        """
        takes a pd.Timestamp() instance and a pd.Series with dates in it
        calcs the delta between `timepoint` and each date in `time_series`
        returns the closest date and optionally the number of days in its time delta
        """
        deltas = np.abs(time_series - timepoint)
        idx_closest_date = np.argmin(deltas)
        res = {"closest_date": time_series.ix[idx_closest_date]}
        idx = ['closest_date']
        if add_time_delta_column:
            res["closest_delta"] = deltas[idx_closest_date]
            idx.append('closest_delta')
        return pd.Series(res, index=idx)


    def get_weather_df(self, w_dbname, w_tbname):
        logger.info("Getting weather from: {st} -- {en} ".format(st=self.start_datetime,
                                                                 en=self.end_datetime))
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
        try:
            # update the database
            combined_df = df
            combined_df.rename(columns={'timestamp': 'time'}, inplace=True)
            logger.info("Updating database")
            #for ind, row in combined_df.iterrows():
            records = combined_df[['time', 'VisibilityMPH', 'Conditions']].to_dict(orient='records')
            """
            Files = ToFilesDB
            query_obj = session.query(Files)
            logger.info("starting loop")
            for record in records:
                logger.info("updating: "+str(record['time']))
                query_obj.filter(Files.timestamp == record['time']).\
                    update({'visibility': record['VisibilityMPH'], 
                            'cloud': record['Conditions']})
            """
            stmt = update(table).\
                where(table.c.timestamp == bindparam('time')).\
                values({'visibility' : bindparam('VisibilityMPH'), 
                        'conditions' : bindparam('Conditions')})
            session.execute(stmt, records)
            """
            #upd = update(table) \
            #.where(table.c.timestamp == row['timestamp']) \
            #.values({'visibility': row['VisibilityMPH'],
            #'conditions'   : row['Conditions']})

            #session.execute(upd)
            """
            logger.info("Commiting to Database")
            session.commit()
        except exc.IntegrityError:
            logger.warning("Duplicate Entry in updating weather "+str(os.path.basename(f)))
            session.rollback()
