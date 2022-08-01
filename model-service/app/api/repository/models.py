from gettext import translation

import sqlalchemy
from .database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import TIME, Column, Float, ForeignKey, Integer, String, TIMESTAMP

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
Test schema.
'''
class Test(Base):
    __tablename__ = 'tests'
    id = Column(Integer, primary_key=True, index=True)
    translation_id = Column(Integer, ForeignKey('translation.id'))

    translation = relationship('Translation', back_populates='tests')

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
    tests = relationship('Test', back_populates='translation')
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


'''
Schema of a single labour market Learning Unit
'''
class LearningUnitLabourMarket(Base):
    __tablename__ = 'learning_unit_labour_market'
    id = Column(Integer, primary_key=True, index=True)


'''
Schema of a single user's completed tests
'''
class UserTest(Base):
    __tablename__ = 'user_test_tracker'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    test_id = Column(Integer, ForeignKey('tests.id'))
    accuracy = Column(Float)
    submitted_on = Column(sqlalchemy.TIMESTAMP)
    used_for_recap_test = Column(Integer)

    test = relationship('Test')
    user = relationship('User', back_populates='user_test')


'''
Schema of a single user's completed Learning Units and labour market Learning Units
liked can be 1 or -1 (if user has completed the LU). Otherwise is 0.
'''
class UserLearningUnit(Base):
    __tablename__ = 'user_learning_unit_tracker'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    learning_unit_id = Column(Integer, ForeignKey('learning_unit.id'))
    learning_unit_labour_market_id = Column(Integer, ForeignKey('learning_unit_labour_market.id'))
    liked = Column(Integer)
    test_completed_on = Column(TIMESTAMP)
    completed_on = Column(TIMESTAMP)

    learning_unit = relationship('LearningUnit')
    user = relationship('User', back_populates='user_learning_unit')

'''
Schema of a single user's preferred clusters and eqf levels for each cluster
'''
class UserClusterSkill(Base):
    __tablename__ = 'user_cluster_skill'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    cluster_id = Column(String)
    skill_value = Column(String)
    use_for_startup = Column(Integer)

    user = relationship('User', back_populates='user_cluster_skill')



'''
Schema of a single User
'''
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    user_cluster_skill = relationship('UserClusterSkill', back_populates='user')
    user_learning_unit = relationship('UserLearningUnit', back_populates='user')
    user_test = relationship('UserTest', back_populates='user')

    