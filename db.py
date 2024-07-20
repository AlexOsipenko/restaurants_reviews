import os
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ClassificationResult(Base):
    __tablename__ = 'classification_results'

    id = Column(Integer, primary_key=True)
    review_text = Column(String)
    predicted_rating = Column(Float)

db_path = os.path.join(os.path.dirname(__file__), 'reviews.db')

engine = create_engine(f'sqlite:///{db_path}')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

session = Session()