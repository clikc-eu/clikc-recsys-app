import datetime
from fastapi import HTTPException, status
from typing import List
from .engine import training, prediction
from .schemas import RecommendOut, StatusOut, StatusTrainingOut
from .constants import TrainingJob
from app import main
from apscheduler.schedulers import base
from sqlalchemy.orm import Session



class ModelService():

    def get_model_status(self):
        return StatusOut(server_on=True, model_trained=training.check_trained_model(), random_mode_on=main.random_mode)

    def train_model(self):

        if main.random_mode == True:
            return StatusTrainingOut(training_completed=False)

        # main.scheduler.get_job(job_id=TrainingJob.JOB_ID).modify(
        #     next_run_time=datetime.datetime.now())

        training.train_model()

        return StatusTrainingOut(training_completed=True)

    def get_recommendations_for_user(self, user_id: int, db: Session):
        is_labour_market, ids = prediction.predict_for_user(user_id=user_id, random_mode=main.random_mode, db=db)
        return RecommendOut(is_labour_market=is_labour_market, ids=ids)