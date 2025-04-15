"""Database handler"""
import os
from sqlalchemy import create_engine, and_, MetaData, ForeignKey, Table, Column, Integer, String, Float, select, update, func, delete, UniqueConstraint
from sqlalchemy.orm import Session
from sqlalchemy.sql import functions
from sqlalchemy.dialects.postgresql import UUID
import uuid
import pandas as pd
from string import ascii_uppercase
import warnings



class Database:
    def __init__(self):
        """
        Table Hierarchy:
            experimentdata - describes experiment
            welldata - describes wells
            channeldata - describes excitation and exposure per channel
            tiledata - describes tiles
            celldata - describes cells
            punctadata - describes puncta within cells
            organelledata - describes organelles within cells


            experimentdata <->  welldata  <-> channeldata
                                    | 
                                    |  
                                tiledata
                                    |
                                    |
                                celldata
                                /       \
                               /         \                                                                     eccentricity, perimeter, extent, solidity

                        punctadata    organelledata

        """
        # _df = pd.read_csv('/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv')

        # Path to the CSV file
        file_path = '/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv'

        # Check if the file exists
        if os.path.exists(file_path):
            try:
                _df = pd.read_csv(file_path)
                print("CSV file read successfully:")
                print(_df.head())  # Print the first few rows of the dataframe
            except Exception as e:
                print(f"An error occurred while reading the file: {e}")
        else:
            print(f"File '{file_path}' does not exist.")
        pw = _df.pw.iloc[0]
        conn_string = f'postgresql://postgres:{pw}@fb-postgres01.gladstone.internal:5432/galaxy'
        # url_object = URL.create(
        #     drivername="postgresql",
        #     username="postgres",
        #     password=f"{pw}",  # plain (unescaped) text
        #     host="fb-postgres01.gladstone.internal",
        #     port='5432',
        #     database="galaxy"
        # )
        self.engine = create_engine(conn_string, future=True, pool_size=10,
                                      max_overflow=2,
                                      pool_recycle=300,
                                      pool_pre_ping=True,
                                      pool_use_lifo=True)
        self.meta = MetaData()
        self.meta.reflect(bind=self.engine)

        with self.engine.connect() as connection:
            print('Connected to Database celldata.')
            if not self.engine.dialect.has_table(connection, 'experimentdata'):
                self.create_experimentdata_table()
            if not self.engine.dialect.has_table(connection, 'welldata'):
                self.create_welldata_table()
            if not self.engine.dialect.has_table(connection, 'channeldata'):
                self.create_channeldata_table()
            if not self.engine.dialect.has_table(connection, 'tiledata'):
                self.create_tiledata_table()
            if not self.engine.dialect.has_table(connection, 'celldata'):
                self.create_celldata_table()
            if not self.engine.dialect.has_table(connection, 'dosagedata'):
                self.create_dosagedata_table()
            if not self.engine.dialect.has_table(connection, 'intensitycelldata'):
                self.create_intensitycelldata_table()
            if not self.engine.dialect.has_table(connection, 'punctadata'):
                self.create_punctadata_table()
            if not self.engine.dialect.has_table(connection, 'intensitypunctadata'):
                self.create_intensitypunctadata_table()
            if not self.engine.dialect.has_table(connection, 'organelledata'):
                self.create_organelledata_table()
            if not self.engine.dialect.has_table(connection, 'cropdata'):
                self.create_cropdata_table()
            if not self.engine.dialect.has_table(connection, 'modeldata'):
                self.create_modeldata_table()
            if not self.engine.dialect.has_table(connection, 'modelcropdata'):
                self.create_modelcropdata_table()
    def create_experimentdata_table(self):
        # TODO: default uuid4 isn't working through sqlalchemy so I added the default with SQL
        # ALTER TABLE experimentdata
        # ALTER id SET DEFAULT uuid_generate_v4();
        experimentdata = Table(
            'experimentdata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column('experiment', String),
            Column('researcher', String),
            Column('description', String),
            Column('project', String),
            Column('platetype', String),
            Column('wellcount', Integer),
            Column('imagedir', String),
            Column('analysisdir', String),
            UniqueConstraint('experiment', name='unique_experiment')
        )
        self.meta.create_all(self.engine)

    def create_welldata_table(self):
        welldata = Table(
            'welldata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('well', String),
            Column('celltype', String),
            Column('condition', String)
        )
        self.meta.create_all(self.engine)

    def create_channeldata_table(self):
        channeldata = Table(
            'channeldata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("welldata_id", UUID(as_uuid=True), ForeignKey("welldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('channel', String),
            Column('objective', String),
            Column('exposure', Float),
            Column('violet', Float),
            Column('blue', Float),
            Column('cyan', Float),
            Column('teal', Float),
            Column('green', Float),
            Column('yellow', Float),
            Column('red', Float),
            Column('nir', Float),
            Column('405nm-5', Float),
            Column('447nm-6', Float),
            Column('488nm-7', Float),
            Column('516nm-2', Float),
            Column('561nm-4', Float),
            Column('642nm-3', Float),
            Column('cobolt_intensity', Float)
        )
        self.meta.create_all(self.engine)

    def create_tiledata_table(self):
        tiledata = Table(
            'tiledata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("welldata_id", UUID(as_uuid=True), ForeignKey("welldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("channeldata_id", UUID(as_uuid=True), ForeignKey("channeldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('tile', Integer),
            Column('pid', String),
            Column('hours', Float),
            Column('timepoint', Integer),
            Column('overlap', Float),
            Column('zstep', Integer),
            Column('zstep_size', Float),
            Column('filename', String),
            Column('time_imaged', String),
            Column('maskpath', String),
            Column('trackedmaskpath', String),
        )
        self.meta.create_all(self.engine)

    def create_celldata_table(self):
        celldata = Table(
            'celldata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("welldata_id", UUID(as_uuid=True), ForeignKey("welldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("tiledata_id", UUID(as_uuid=True), ForeignKey("tiledata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('cellid', Integer),
            Column('randomcellid', Integer),
            Column('centroid_x', Float),
            Column('centroid_y', Float),
            Column('area', Float),
            Column('solidity', Float),
            Column('extent', Float),
            Column('perimeter', Float),
            Column('eccentricity', Float),
            Column('axis_major_length', Float),
            Column('axis_minor_length', Float),
        )
        self.meta.create_all(self.engine)

    def create_cropdata_table(self):
        cropdata = Table(
            'cropdata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("welldata_id", UUID(as_uuid=True), ForeignKey("welldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("channeldata_id", UUID(as_uuid=True), ForeignKey("channeldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("celldata_id", UUID(as_uuid=True), ForeignKey("celldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('croppath', String)
        )
        self.meta.create_all(self.engine)

    def create_modeldata_table(self):
        modeldata = Table(
            'modeldata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('modelname', String),
            Column('modelpath', String),
            Column('wandbpath', String),
            Column('train_loss', Float),
            Column('val_loss', Float),
            Column('train_acc', Float),
            Column('val_acc', Float),

        )
        self.meta.create_all(self.engine)

    def create_modelcropdata_table(self):
        """Model crop data could include multiple crops. Expect just one celldata id though. 
        Model for puncta will need a new table"""
        modelcropdata = Table(
            'modelcropdata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("model_id", UUID(as_uuid=True), ForeignKey("modeldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("welldata_id", UUID(as_uuid=True), ForeignKey("welldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("celldata_id", UUID(as_uuid=True), ForeignKey("celldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('stage', String),
            Column('output', Float),
            Column('prediction', Float),
            Column('groundtruth', Float),
            Column('prediction_label', String),
            Column('groundtruth_label', String),

        )
        self.meta.create_all(self.engine)

    def create_intensitycelldata_table(self):
        intensitycelldata = Table(
            'intensitycelldata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("welldata_id", UUID(as_uuid=True), ForeignKey("welldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("tiledata_id", UUID(as_uuid=True), ForeignKey("tiledata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("celldata_id", UUID(as_uuid=True), ForeignKey("celldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("channeldata_id", UUID(as_uuid=True), ForeignKey("channeldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('intensity_max', Float),
            Column('intensity_mean', Float),
            Column('intensity_min', Float),
            Column('intensity_std', Float),
        )
        self.meta.create_all(self.engine)

    def create_intensitypunctadata_table(self):
        intensitypunctadata = Table(
            'intensitypunctadata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("welldata_id", UUID(as_uuid=True), ForeignKey("welldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("tiledata_id", UUID(as_uuid=True), ForeignKey("tiledata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("celldata_id", UUID(as_uuid=True), ForeignKey("celldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("punctadata_id", UUID(as_uuid=True), ForeignKey("punctadata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("channeldata_id", UUID(as_uuid=True), ForeignKey("channeldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('intensity_max', Float),
            Column('intensity_mean', Float),
            Column('intensity_min', Float),
            Column('intensity_std', Float),
        )
        self.meta.create_all(self.engine)
    
    def create_dosagedata_table(self):
        dosagedata = Table(
            'dosagedata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("welldata_id", UUID(as_uuid=True), ForeignKey("welldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('name', String),
            Column('dosage', Float),
            Column('kind', String)
        )
        self.meta.create_all(self.engine)

    def create_punctadata_table(self):
        punctadata = Table(
            'punctadata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("welldata_id", UUID(as_uuid=True), ForeignKey("welldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("tiledata_id", UUID(as_uuid=True), ForeignKey("tiledata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("celldata_id", UUID(as_uuid=True), ForeignKey("celldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('punctaid', Integer),
            Column('randompunctaid', Integer),
            Column('centroid_x', Float),
            Column('centroid_y', Float),
            Column('area', Float),
            Column('solidity', Float),
            Column('extent', Float),
            Column('perimeter', Float),
            Column('eccentricity', Float),
            Column('axis_major_length', Float),
            Column('axis_minor_length', Float),
        )
        self.meta.create_all(self.engine)

    def create_organelledata_table(self):
        organelledata = Table(
            'organelledata', self.meta,
            Column('id', UUID(as_uuid=True), default=uuid.uuid4, primary_key=True),
            Column("experimentdata_id", UUID(as_uuid=True), ForeignKey("experimentdata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column("celldata_id", UUID(as_uuid=True), ForeignKey("celldata.id", ondelete="CASCADE"),
                   index=True, nullable=False),
            Column('organelleid', Integer),
            Column('randomorganelleid', Integer),
            Column('organellename', String),
            Column('centroid_x', Float),
            Column('centroid_y', Float),
            Column('area', Float),
            Column('axis_major_length', Float),
            Column('axis_minor_length', Float),
        )
        self.meta.create_all(self.engine)

    def add_row(self, tablename: str, dct: [dict,list]):
        with self.engine.connect() as connection:
            ins = self.meta.tables[tablename].insert().values(dct)
            db = connection.execute(ins)
            connection.commit()

    def update(self, tablename: str, update_dct: dict, kwargs):
        with self.engine.connect() as connection:
            ins = update(self.meta.tables[tablename]).filter_by(**kwargs).values(update_dct)
            db = connection.execute(ins)
            connection.commit()
            
    def update_prefix_path(self, tablename:str,exp_uuid, old_string:str, new_string:str):
        with self.engine.connect() as connection:
            update_stmt1 = (
                update(self.meta.tables[tablename]).
                where(and_(
                    self.meta.tables[tablename].c.experimentdata_id==exp_uuid,
                    self.meta.tables[tablename].c.filename.contains(old_string))).
                values(filename=func.replace(self.meta.tables[tablename].c.filename, old_string, new_string))
            )
            print('stmt1', update_stmt1)
            connection.execute(update_stmt1)
            update_stmt2 = (
                update(self.meta.tables[tablename]).
                where(and_(
                    self.meta.tables[tablename].c.experimentdata_id==exp_uuid,
                    self.meta.tables[tablename].c.maskpath.contains(old_string))).
                values(maskpath=func.replace(self.meta.tables[tablename].c.maskpath, old_string, new_string))
            )
            connection.execute(update_stmt2)
            update_stmt3 = (
                update(self.meta.tables[tablename]).
                where(and_(
                    self.meta.tables[tablename].c.experimentdata_id==exp_uuid,
                    self.meta.tables[tablename].c.trackedmaskpath.contains(old_string))).
                values(trackedmaskpath=func.replace(self.meta.tables[tablename].c.trackedmaskpath, old_string, new_string))
            )
            connection.execute(update_stmt3)
            connection.commit()
        
            
    def update_slashes(self, tablename: str, exp_uuid):
        """Assumes tiledata, won't work for experimentdata due to primary key name"""
        with self.engine.connect() as connection:
            update_stmt = (
                update(self.meta.tables[tablename]).
                where(self.meta.tables[tablename].c.experimentdata_id==exp_uuid).
                values(filename=func.replace(self.meta.tables[tablename].c.filename, '\\', '/'),
                       maskpath=func.replace(self.meta.tables[tablename].c.maskpath, '\\', '/'),
                       trackedmaskpath=func.replace(self.meta.tables[tablename].c.trackedmaskpath, '\\', '/'))
        )
            db = connection.execute(update_stmt)
            connection.commit()

    def get_table_uuid(self, tablename: str, kwargs):
        with self.engine.connect() as connection:
            stmt = select(self.meta.tables[tablename].c['id']). \
                filter_by(**kwargs)
            result = connection.execute(stmt).all()
        if len(result) == 0:
            return None
        return result[0][0]
    
    
    def get_table_analysisdir(self, tablename: str, kwargs):
        with self.engine.connect() as connection:
            stmt = select(self.meta.tables[tablename].c['analysisdir']). \
                filter_by(**kwargs)
            result = connection.execute(stmt).all()
        if len(result) == 0:
            return None
        return result[0][0]
    
    def get_table_value(self, tablename: str, column: str, kwargs):
        with self.engine.connect() as connection:
            stmt = select(self.meta.tables[tablename].c[column]). \
                filter_by(**kwargs)
            result = connection.execute(stmt).all()
        if len(result) == 0:
            return None
        return result

    def get_table_cols(self, tablename: str, columns: list, kwargs):
        with self.engine.connect() as connection:
            stmt = select(self.meta.tables[tablename].c[columns[0], columns[1]]). \
                filter_by(**kwargs)
            result = connection.execute(stmt).all()
        if len(result) == 0:
            return None
        return result

    def delete_based_on_duplicate_name(self, tablename: str, kwargs):
        # select the the max value in time imaged

        with Session(self.engine) as session:
            qry = session.query(self.meta.tables[tablename]). \
                filter_by(**kwargs).first()

        if qry is not None:
            print(f'Deleting from database: {kwargs}')
            warnings.warn(f'Deleting from database: {kwargs}')
            with self.engine.connect() as connection:
                stmt = delete(self.meta.tables[tablename]).filter_by(**kwargs)
                res = connection.execute(stmt)
                connection.commit()

    def get_df_from_query(self, tablename, kwargs):
        statement = select(self.meta.tables[tablename]).filter_by(**kwargs)
        with self.engine.connect() as connection:
            df = pd.read_sql(statement, connection)
        return df
    
    def join_table(self, table1, table2):
        with Session(self.engine) as session:
            qry = session.query(self.meta.tables[table1]). \
                join(table2).all()
        return qry