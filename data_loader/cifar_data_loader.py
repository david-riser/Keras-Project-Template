from base.base_data_loader import BaseDataLoader
from utils.factory import create
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CifarDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(CifarDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()
        self.X_train = self.X_train.reshape((-1, 32, 32, 3))
        self.X_test = self.X_test.reshape((-1, 32, 32, 3))

        self.X_train = self.X_train.astype('float') / 255.
        self.X_test = self.X_test.astype('float') / 255.

        if hasattr(self.config.data_loader.toDict(), 'preprocessing_function'):
            self.preprocess = create("tensorflow.keras.applications.{}".format(
                self.config.data_loader.preprocessing_function
            ))
        else:
            self.preprocess = None
            
        self.train_gen = ImageDataGenerator(preprocessing_function=self.preprocess)
        self.train_flow = self.train_gen.flow(
            self.X_train, self.y_train,
            batch_size=self.config.data_loader.batch_size,
        )
        
        self.test_gen = ImageDataGenerator(preprocessing_function=self.preprocess)
        self.test_flow = self.test_gen.flow(
            self.X_test, self.y_test,
            batch_size=self.config.data_loader.batch_size,
        )
        
    def get_train_flow(self):
        return self.train_flow

    def get_test_flow(self):
        return self.test_flow

    def get_test_data(self):
        return self.X_test, self.y_test
