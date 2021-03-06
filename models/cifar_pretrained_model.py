import numpy as np
import tensorflow as tf

from base.base_model import BaseModel
from utils.factory import create
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

class CifarPretrainedModel(BaseModel):
    def __init__(self, config):
        super(CifarPretrainedModel, self).__init__(config)
        self.model_builder = create("tensorflow.keras.applications.{}".format(
            self.config.model.backbone
        ))
        self.build_model()

    def build_model(self):
        """ The model is built with most of the layers frozen. """
        
        self.backbone = self.model_builder(
            weights='imagenet',
            pooling=self.config.model.pooling,
            include_top=False
        )

        x = Dense(2048, activation='relu')(
            self.backbone.output
        )
        outputs = Dense(10, activation='softmax')(x)
        
        self.model = Model(inputs=self.backbone.input, outputs=outputs)
        self.model.compile(
              loss='sparse_categorical_crossentropy',
              optimizer=self.config.model.optimizer,
              metrics=['accuracy']
        )
        
        for layer in self.model.layers[:-2]:
            layer.trainable = False

        for layer in self.model.layers[-2:]:
            layer.trainable = True
