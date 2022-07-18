from .database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, ForeignKey, Integer, String

'''
DynamicField schema.
It contains texts and links to multimedia elements. Here only text fields are used.
'''
class DynamicField(Base):
    __tablename__ = 'dinamic_field'
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String)
    content = Column(String)
    translations_id = Column(Integer, ForeignKey('translation.id'))

    translation = relationship('Translation', back_populates='dynamic_fields')

    

'''
Schema of one of the possible translations of a Learning Unit.
It contains the didactic content.
'''
class Translation(Base):
    __tablename__ = 'translation'
    id = Column(Integer, primary_key=True, index=True)
    language_name = Column(String)  # language code
    title = Column(String)
    subtitle = Column(String)
    keywords = Column(String)       # list of keywords, comma separated
    introduction = Column(String)
    text_area = Column(String)
    learning_unit_id = Column(Integer, ForeignKey('learning_unit.id'))

    dynamic_fields = relationship('DynamicField', back_populates='translation')

    learning_unit = relationship('LearningUnit', back_populates='translations')


'''
Schema of a single Learning Unit
'''
class LearningUnit(Base):
    __tablename__ = 'learning_unit'
    id = Column(Integer, primary_key=True, index=True)
    identifier = Column(String)
    cluster_number = Column(String)
    skill = Column(String)
    eqf_level = Column(String)

    translations = relationship("Translation", back_populates='learning_unit')
