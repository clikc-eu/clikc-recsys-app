from .engine import training
from .models import StatusOut

class ModelService():

    def get_model_status(self):
        return StatusOut(server_on=True, model_trained=training.check_trained_model())
