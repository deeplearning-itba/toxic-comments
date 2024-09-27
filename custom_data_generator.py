import tensorflow as tf
import numpy as np
import math

##
# cdg = CustomDataGenerator()
# model.fit(cdg)

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_sparse, y_true, batch_size, shuffle=True):
        self.X_sparse = X_sparse
        self.y_true = y_true
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return self.X_sparse.shape[0] // self.batch_size
    
    def __getitem__(self, batch_index):
        indexes = self.indexes[batch_index*self.batch_size: (batch_index+1)*self.batch_size]

        X = self.X_sparse[indexes].todense()
        y = self.y_true[indexes]
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.X_sparse.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)
