import os
from sqlalchemy import Sequence
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, PrimaryKeyConstraint
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ToFilesDB(Base):


    __tablename__ = os.getenv("CUIP_TBNAME")
    
    gid           = Column('gid',         Integer)
    fname         = Column('fname',       String(length=100, convert_unicode=True))
    fpath         = Column('fpath',       String(length=100, convert_unicode=True))
    fsize         = Column('fsize',       Integer)
    mean          = Column('mean',        Float)
    std           = Column('std',         Float)
    bright_pix    = Column('bright_pix',  Integer)
    timestamp     = Column('timestamp',   DateTime(timezone=False), index=True)
    visibility    = Column('visibility',  Float)
    cloud         = Column('conditions',  String(length=50, convert_unicode=True))
    temperature   = Column('temperature', Float)
    roffset       = Column('roffset',     Integer)
    coffset       = Column('coffset',     Integer)
    angle         = Column('angle',       Float)
    usable        = Column('usable',      Boolean)
    fnumber       = Column(Integer, Sequence('fnumber'))

    __table_args__ = (PrimaryKeyConstraint('fpath', 'fname'), 
                      {})



class ToWeatherDB(Base):


    __tablename__ = os.getenv("CUIP_WEATHER_TBNAME")
    
    timestamp     = Column('Time',                 DateTime(timezone=False), primary_key=True)
    temperatureF  = Column('TemperatureF',         Float)
    dew_pointF    = Column('Dew PointF',           Float)
    humidity      = Column('Humidity',             Float)
    sea_level_pr  = Column('Sea Level PressureIn', Float)
    visibility    = Column('VisibilityMPH',        Float)
    wind_dir      = Column('Wind Direction',       String(50))
    wind_speed    = Column('Wind SpeedMPH',        Float)
    gust_speed    = Column('Gust SpeedMPH',        Float)
    precipitation = Column('PrecipitationIn',      Float)
    events        = Column('Events',               String(50))
    conditions    = Column('Conditions',           String(50))
    wind_dir_deg  = Column('WindDirDegrees',       Float)
    

def clone_table(where):
    """
    Create a new table based on (multiple) where
    clause(s)
    Parameters
    ----------
    where: dict
        column names as key and the value against which
        database will be filtered
    """
    pass

