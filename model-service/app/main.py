from fastapi import FastAPI
from .api.model import model
from .api.engine import training
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit

# Run training at server start if needed
if training.check_trained_model() == False:
    training.train_model()

# create schedule for printing time - at 01:30
scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(
    func=training.train_model,
    trigger=CronTrigger(hour=1, minute=30),
    id='model_training_job',
    name='Train model each day',
    replace_existing=True)
# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())


app = FastAPI(openapi_url="/api/v1/model/openapi.json", docs_url="/api/v1/model/docs")

app.include_router(model, prefix='/api/v1/model', tags=['model'])
