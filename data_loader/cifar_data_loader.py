from base.base_data_loader import BaseDataLoader
from utils.factory import create
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CifarDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(CifarDataLoader, self).__init__(config)

        if 'preprocessing_function' in self.config.data_loader.toDict():
            self.preprocess_func = create("tensorflow.keras.applications.{}".format(
                self.config.data_loader.preprocessing_function))
        else:
            self.preprocess_func = lambda x: x / 255.

        self.load()
        self.preprocess()
        
    def load(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()
        self.X_train = self.X_train.reshape((-1, 32, 32, 3))
        self.X_test = self.X_test.reshape((-1, 32, 32, 3))
        self.X_train = self.X_train.astype('float')
        self.X_test = self.X_test.astype('float')

        self.train_gen = ImageDataGenerator(preprocessing_function=self.preprocess_func)
        self.train_flow = self.train_gen.flow(
            self.X_train, self.y_train,
            batch_size=self.config.data_loader.batch_size,
        )
        
        self.test_gen = ImageDataGenerator(preprocessing_function=self.preprocess_func)
        self.test_flow = self.test_gen.flow(
            self.X_test, self.y_test,
            batch_size=self.config.data_loader.batch_size,
        )

    def preprocess(self):                    
        if self.preprocess_func:
            print("Applying preprocessing to test set.")
            self.X_test = self.preprocess_func(self.X_test)

            
    def get_train_flow(self):
        return self.train_flow

    def get_test_flow(self):
        return self.test_flow

    def get_test_data(self):
        return (self.X_test, self.y_test)
