"""Low-level PostgreSQL interface for the Galaxy imaging database.

Provides the ``Database`` class which wraps SQLAlchemy to create, query,
and update tables used by the Finkbeiner Lab image-analysis pipeline.
All tables use UUID primary keys and follow a hierarchy rooted at
``experimentdata``.
"""

import os
import uuid
import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy import (
    Column, Float, Integer, MetaData, String, Table, UniqueConstraint,
    ForeignKey, and_, create_engine, delete, func, select, update,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool


class Database:
    """Low-level interface to the Galaxy PostgreSQL database.

    On instantiation, connects to the database, reflects existing schema,
    and creates any missing tables.  Uses ``NullPool`` so each connection
    is opened and closed on demand (safe for multi-process / Slurm usage).

    Table hierarchy::

        experimentdata <-> welldata <-> channeldata
                               |
                           tiledata
                               |
                           celldata
                          /        \\
                    punctadata   organelledata
    """

    CREDENTIALS_PATH: str = '/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv'

    def __init__(self) -> None:
        """Initialise the database connection and ensure all tables exist."""
        file_path = self.CREDENTIALS_PATH
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Database credentials file not found: '{file_path}'. "
                "Cannot connect to the 'galaxy' PostgreSQL database."
            )
        try:
            _df = pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read database credentials from '{file_path}': {e}"
            ) from e
        try:
            pw = _df.pw.iloc[0]
        except (AttributeError, IndexError) as e:
            raise ValueError(
                f"Credentials file '{file_path}' is malformed: "
                "expected a 'pw' column with at least one row."
            ) from e
        # Connection targets default to the lab cluster but can be overridden
        # via environment variables (e.g. when running on a standalone VM or
        # against a least-privilege role instead of the postgres superuser).
        db_user = os.environ.get('GALAXY_DB_USER', 'postgres')
        db_host = os.environ.get('GALAXY_DB_HOST', 'fb-postgres01.gladstone.internal')
        db_port = os.environ.get('GALAXY_DB_PORT', '5432')
        db_name = os.environ.get('GALAXY_DB_NAME', 'galaxy')
        conn_string = f'postgresql://{db_user}:{pw}@{db_host}:{db_port}/{db_name}'

        self.engine = create_engine(conn_string, poolclass=NullPool)
        self.meta = MetaData()
        self.meta.reflect(bind=self.engine)

        with self.engine.connect() as connection:
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
            self.meta.reflect(bind=self.engine, resolve_fks=True)

    def create_experimentdata_table(self) -> None:
        """Create the top-level experiment metadata table."""
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

    def create_welldata_table(self) -> None:
        """Create the well-level metadata table (FK to experimentdata)."""
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

    def create_channeldata_table(self) -> None:
        """Create the channel metadata table (excitation wavelengths, exposure)."""
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

    def create_tiledata_table(self) -> None:
        """Create the tile-level image data table (filenames, masks, timepoints)."""
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
            Column('newtrackedmontage', String),
        )
        self.meta.create_all(self.engine)

    def create_celldata_table(self) -> None:
        """Create the cell-level morphology data table."""
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

    def create_cropdata_table(self) -> None:
        """Create the crop data table (cell image crops per channel)."""
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

    def create_modeldata_table(self) -> None:
        """Create the ML model metadata table."""
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

    def create_modelcropdata_table(self) -> None:
        """Create the model crop data table.

        May include multiple crops but expects one celldata_id per row.
        A separate table would be needed for puncta-level model data.
        """
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

    def create_intensitycelldata_table(self) -> None:
        """Create the per-cell intensity statistics table."""
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

    def create_intensitypunctadata_table(self) -> None:
        """Create the per-puncta intensity statistics table."""
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
    
    def create_dosagedata_table(self) -> None:
        """Create the dosage/treatment data table."""
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

    def create_punctadata_table(self) -> None:
        """Create the puncta-level morphology data table."""
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

    def create_organelledata_table(self) -> None:
        """Create the organelle-level morphology data table."""
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

    def add_row(self, tablename: str, dct: Union[Dict[str, Any], List[Dict[str, Any]]],
                chunk_size: int = 5000) -> None:
        """Insert one or more rows into a table.

        Bulk inserts are chunked so a single very large multi-VALUES statement
        cannot exceed PostgreSQL's bound-parameter limit or blow up statement
        memory (celldata/intensitycelldata can be thousands of rows per well).

        Args:
            tablename: Name of the target table.
            dct: A single dict (one row) or list of dicts (bulk insert).
            chunk_size: Max rows per INSERT statement for the bulk-list case.
        """
        table = self.meta.tables[tablename]
        with self.engine.connect() as connection:
            if isinstance(dct, list):
                if not dct:
                    return
                for start in range(0, len(dct), chunk_size):
                    chunk = dct[start:start + chunk_size]
                    connection.execute(table.insert().values(chunk))
            else:
                connection.execute(table.insert().values(dct))
            connection.commit()

    def update(self, tablename: str, update_dct: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        """Update rows matching ``kwargs`` with the values in ``update_dct``.

        Args:
            tablename: Name of the target table.
            update_dct: Column-value pairs to set.
            kwargs: Filter criteria passed to ``filter_by``.
        """
        with self.engine.connect() as connection:
            stmt = update(self.meta.tables[tablename]).filter_by(**kwargs).values(update_dct)
            connection.execute(stmt)
            connection.commit()
            
    def update_prefix_path(self, tablename: str, exp_uuid: Any, old_string: str, new_string: str) -> None:
        """Replace a path prefix in filename, maskpath, and trackedmaskpath columns.

        Args:
            tablename: Table to update (typically 'tiledata').
            exp_uuid: UUID of the experiment to scope the update.
            old_string: Substring to find in path columns.
            new_string: Replacement substring.
        """
        with self.engine.connect() as connection:
            # Update filename column
            update_stmt1 = (
                update(self.meta.tables[tablename]).
                where(and_(
                    self.meta.tables[tablename].c.experimentdata_id==exp_uuid,
                    self.meta.tables[tablename].c.filename.contains(old_string))).
                values(filename=func.replace(self.meta.tables[tablename].c.filename, old_string, new_string))
            )
            connection.execute(update_stmt1)
            # Update maskpath column
            update_stmt2 = (
                update(self.meta.tables[tablename]).
                where(and_(
                    self.meta.tables[tablename].c.experimentdata_id==exp_uuid,
                    self.meta.tables[tablename].c.maskpath.contains(old_string))).
                values(maskpath=func.replace(self.meta.tables[tablename].c.maskpath, old_string, new_string))
            )
            connection.execute(update_stmt2)
            # Update trackedmaskpath column
            update_stmt3 = (
                update(self.meta.tables[tablename]).
                where(and_(
                    self.meta.tables[tablename].c.experimentdata_id==exp_uuid,
                    self.meta.tables[tablename].c.trackedmaskpath.contains(old_string))).
                values(trackedmaskpath=func.replace(self.meta.tables[tablename].c.trackedmaskpath, old_string, new_string))
            )
            connection.execute(update_stmt3)
            connection.commit()

    def update_slashes(self, tablename: str, exp_uuid: Any) -> None:
        """Replace backslashes with forward slashes in all path columns.

        Operates on filename, maskpath, and trackedmaskpath for a given
        experiment.  Assumes the table has an ``experimentdata_id`` column
        (will not work on ``experimentdata`` itself).

        Args:
            tablename: Table to update (typically 'tiledata').
            exp_uuid: UUID of the experiment to scope the update.
        """
        with self.engine.connect() as connection:
            update_stmt = (
                update(self.meta.tables[tablename]).
                where(self.meta.tables[tablename].c.experimentdata_id==exp_uuid).
                values(filename=func.replace(self.meta.tables[tablename].c.filename, '\\', '/'),
                       maskpath=func.replace(self.meta.tables[tablename].c.maskpath, '\\', '/'),
                       trackedmaskpath=func.replace(self.meta.tables[tablename].c.trackedmaskpath, '\\', '/'))
        )
            connection.execute(update_stmt)
            connection.commit()

    def get_table_uuid(self, tablename: str, kwargs: Dict[str, Any]) -> Optional[Any]:
        """Return the UUID (``id`` column) of the first matching row.

        Args:
            tablename: Table to query.
            kwargs: Filter criteria passed to ``filter_by``.

        Returns:
            The UUID value, or ``None`` if no row matches.
        """
        with self.engine.connect() as connection:
            stmt = select(self.meta.tables[tablename].c['id']). \
                filter_by(**kwargs)
            result = connection.execute(stmt).all()
        if len(result) == 0:
            return None
        return result[0][0]

    def get_table_analysisdir(self, tablename: str, kwargs: Dict[str, Any]) -> Optional[str]:
        """Return the ``analysisdir`` value for the first matching row.

        Args:
            tablename: Table to query (typically 'experimentdata').
            kwargs: Filter criteria passed to ``filter_by``.

        Returns:
            The analysis directory path, or ``None`` if no row matches.
        """
        with self.engine.connect() as connection:
            stmt = select(self.meta.tables[tablename].c['analysisdir']). \
                filter_by(**kwargs)
            result = connection.execute(stmt).all()
        if len(result) == 0:
            return None
        return result[0][0]

    def get_table_value(self, tablename: str, column: str, kwargs: Dict[str, Any]) -> Optional[List[Any]]:
        """Return all values of a single column for matching rows.

        Args:
            tablename: Table to query.
            column: Column name to select.
            kwargs: Filter criteria passed to ``filter_by``.

        Returns:
            List of result tuples, or ``None`` if no rows match.
        """
        with self.engine.connect() as connection:
            stmt = select(self.meta.tables[tablename].c[column]). \
                filter_by(**kwargs)
            result = connection.execute(stmt).all()
        if len(result) == 0:
            return None
        return result

    def get_table_cols(self, tablename: str, columns: List[str], kwargs: Dict[str, Any]) -> Optional[List[Any]]:
        """Return values of two columns for matching rows.

        Args:
            tablename: Table to query.
            columns: List of exactly two column names to select.
            kwargs: Filter criteria passed to ``filter_by``.

        Returns:
            List of result tuples, or ``None`` if no rows match.
        """
        with self.engine.connect() as connection:
            stmt = select(self.meta.tables[tablename].c[columns[0], columns[1]]). \
                filter_by(**kwargs)
            result = connection.execute(stmt).all()
        if len(result) == 0:
            return None
        return result

    def delete_based_on_duplicate_name(self, tablename: str, kwargs: Dict[str, Any]) -> None:
        """Delete all rows matching ``kwargs`` if at least one exists.

        Used to remove duplicate entries before re-inserting updated data.

        Args:
            tablename: Table to delete from.
            kwargs: Filter criteria identifying the duplicate rows.
        """
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

    def get_df_from_query(self, tablename: str, kwargs: Dict[str, Any]) -> pd.DataFrame:
        """Return a DataFrame of all rows matching the filter criteria.

        Args:
            tablename: Table to query.
            kwargs: Filter criteria passed to ``filter_by``.

        Returns:
            DataFrame with one row per matching database row.
        """
        statement = select(self.meta.tables[tablename]).filter_by(**kwargs)
        with self.engine.connect() as connection:
            df = pd.read_sql(statement, connection)
        return df

    def join_table(self, table1: str, table2: str) -> List[Any]:
        """Perform a natural join between two tables and return all rows.

        Args:
            table1: Name of the primary table.
            table2: Name of the table to join.

        Returns:
            List of joined row objects.
        """
        with Session(self.engine) as session:
            qry = session.query(self.meta.tables[table1]).join(table2).all()
        return qry