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

    def __init__(self, start_datetime, end_datetime):
        """
        Update database with the contents from Dataframe
        Parameters
        ----------
        start_datetime: datetime.datetime
        end_datetime  : datetime.datetime
        """
        self.start_datetime = start_datetime
        self.end_datetime   = end_datetime
        self.f_dbname       = os.getenv("CUIP_DBNAME")
        self.f_tbname       = os.getenv("CUIP_TBNAME")
        self.w_dbname        = os.getenv("CUIP_WEATHER_DBNAME")
        self.w_tbname       = os.getenv("CUIP_WEATHER_TBNAME")

    def __call__(self, session=None, engine=None, table=None):

        # load the data chunk from files database for similar time
        query = """SELECT * FROM {table} \
                 WHERE "timestamp" \
                 BETWEEN '{first}' AND '{last}'""".\
            format(
                     table=self.f_tbname,
                     first=self.start_datetime,
                     last=self.end_datetime)

        # load result in a dataframe
        dbf   = pd.read_sql(query, engine)
        # sort the files database
        dbf.sort_values(by='timestamp', inplace=True)
        
        # get weather dataframe
        wdf = self.get_weather_df(self.start_datetime,
                                  self.end_datetime)        
        # sort the weather dataframe
        wdf.sort_values(by='Time', inplace=True)
        
        # combine the weather with the chunk
        try:
            logger.info("Adding weather data to {0}".\
                            format(self.f_tbname))

            if not (dbf.empty or wdf.empty):
            # -- for multiprocessing... TBD    
            #    dbf[['closest']] = dbf.timestamp.apply(
            #        self.find_closest_date, args=[wdf.Time])
            #    logger.info("merging")
            #    combined_df = pd.merge(dbf, wdf, left_on=['closest'], right_on=['Time'])
                combined_df = pd.\
                    merge_asof(dbf,
                               wdf[['Time', 'VisibilityMPH', 'Conditions', 'TemperatureF']],
                               left_on='timestamp', right_on='Time',
                               #tolerance=pd.Timedelta('3hours')
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

    def get_weather_df(self, start_datetime, end_datetime):
        logger.info("Getting weather from: {st} -- {en} ".\
                        format(st=start_datetime,
                               en=end_datetime))
        # fir up another engine for weather database
        w_engine = create_engine('postgresql:///{0}'.\
                                     format(self.w_dbname))
        w_md     = MetaData(bind=w_engine)
        w_md.reflect()
        w_table  = Table(self.w_tbname, w_md, autoload=True)
        query    = """SELECT * FROM {table} \
                      WHERE "Time" \
                      BETWEEN '{first}' AND '{last}'""".\
            format(table=self.w_tbname,
                   first=start_datetime,
                   last=end_datetime)
        # load result in a dataframe
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
            logger.info("Updating database")
            # update the database
            combined_df = df
            combined_df.rename(columns={'timestamp': 'time'}, 
                               inplace=True)

            # convert dataframe to dictionary to perform bulk update
            records = combined_df[['time', 'VisibilityMPH', 
                                   'Conditions', 'TemperatureF']].\
                                   to_dict(orient='records')

            # create an orm statement
            stmt = update(table).\
                where(table.c.timestamp == bindparam('time')).\
                values({'visibility' : bindparam('VisibilityMPH'), 
                        'conditions' : bindparam('Conditions'),
                        'temperature': bindparam('TemperatureF')})

            # perform bulk update
            session.execute(stmt, records)

            # commit to database
            session.commit()
        except exc.IntegrityError:
            logger.warning("Possibly duplicate entry"+
                           "in updating weather "+
                           str(os.path.basename(f))+
                           "rolling back database")
            session.rollback()
