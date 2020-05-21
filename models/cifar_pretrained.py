from base.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

class CifarPretrainedModel(BaseModel):
    def __init__(self, config):
        super(CifarPretrainedModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = None
        
        self.model.compile(
              loss='sparse_categorical_crossentropy',
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])
