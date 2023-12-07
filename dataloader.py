import numpy as np
import tensorflow as tf

class DataLoader(object):
    def __init__(self, system, batch_size, batch_size_zero, to_tensor=True):
        self.num_x = system.num_x
        self.range_x = system.range_x
            
        self.batch_size = batch_size
        self.batch_size_zero = batch_size_zero
        self.to_tensor = to_tensor
        
        
    def get_random_input(self):
        x = np.array([])
        
        for i in range(self.num_x):
            x_gen = np.random.uniform(low=self.range_x[f'x{i+1}']['min'], high=self.range_x[f'x{i+1}']['max'], size=self.batch_size)
            x_gen = x_gen.reshape(-1, 1)
            if len(x) == 0:
                x = x_gen.copy()
            else:
                x = np.concatenate((x, x_gen.copy()), axis=-1)
                
        return x
        
        
    def get_zero_input(self):
        x = np.zeros((self.batch_size_zero, self.num_x))
        
        return x
    
    def __call__(self):
        if self.to_tensor:
            return tf.convert_to_tensor(self.get_random_input(), dtype=tf.float32), tf.convert_to_tensor(self.get_zero_input(), dtype=tf.float32)
        else:
            return self.get_random_input(), self.get_zero_input()