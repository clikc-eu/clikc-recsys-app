from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ..util.db_utils import get_db_params

# SQLALCHEMY DATABASE CONNECTION AND CONFIGURATION

# Get DB params from
# 'configuration.json' file
db_name, db_user, db_pw, db_ip, db_port = get_db_params()

SQLALCHEMY_DATABASE_URL = f'mariadb+pymysql://{db_user}:{db_pw}@{db_ip}:{db_port}/{db_name}'

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
