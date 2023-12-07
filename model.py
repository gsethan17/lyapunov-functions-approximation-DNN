import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

def get_model(num_input, num_output, num_layers, num_nodes):
    model = Sequential()
    model.add(Input((num_input,)))
    for i in range(num_layers-1):
        model.add(Dense(num_nodes, activation='tanh'))
    model.add(Dense(num_output))
    
    return model


if __name__ == "__main__":
    model = get_model(
        num_input=2,
        num_output=1,
        num_layers=4,
        num_nodes=64,
    )
    model.summary()
