import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

def get_model(num_input, num_output, num_layers, num_nodes,
            use_bias=True, use_last_bias=True):
    model = Sequential()
    model.add(Input((num_input,)))
    for i in range(num_layers-1):
        model.add(Dense(num_nodes, 
                        use_bias=use_bias,
                        activation='tanh'))
    model.add(Dense(num_output, use_bias=use_last_bias))
    
    return model


if __name__ == "__main__":
    model = get_model(
        num_input=2,
        num_output=1,
        num_layers=4,
        num_nodes=64,
        # use_bias=False,
        use_last_bias=False,
    )
    model.summary()
