from fastapi import FastAPI
from .api.model import model
from .api.engine import training
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit
from .api.constants import TrainingJob
from .api.util.random_mode_utils import check_random_mode
from .api.util.logger import logger

# Check if random recommendations mode has been enabled in
# 'configuration.json' file
random_mode = check_random_mode()

logger.info(f"Random Mode: {random_mode}")

# Run training at server start if needed
if training.check_trained_model() == False and random_mode == False:
    training.train_model()


if random_mode == False:
    # create schedule for printing time - at 01:30
    scheduler = BackgroundScheduler()
    scheduler.start()
    scheduler.add_job(
        func=training.train_model,
        trigger=CronTrigger(hour=1, minute=30),
        id=TrainingJob.JOB_ID,
        name=TrainingJob.JOB_NAME,
        replace_existing=True)
    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

# Uncomment to enable the auto-documentation page
# app = FastAPI(openapi_url="/api/v1/model/openapi.json", docs_url="/api/v1/model/docs")
app = FastAPI(openapi_url=None, docs_url=None)

app.include_router(model, prefix='/api/v1/model', tags=['model'])
