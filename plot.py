import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def draw_3d(model, system, save_path=False):
    X1 = np.linspace(system.range_x['x1']['min'], system.range_x['x1']['max'], 100)
    X2 = np.linspace(system.range_x['x2']['min'], system.range_x['x2']['max'], 100)
    X1, X2 = np.meshgrid(X1, X2)
    
    v_hat = np.zeros(X1.shape)
    v_dt_hat = np.zeros(X1.shape)
    
    for i in range(X1.shape[0]):
        x1_input = tf.convert_to_tensor(X1[i].reshape(-1, 1), dtype=tf.float32)
        x2_input = tf.convert_to_tensor(X2[i].reshape(-1, 1), dtype=tf.float32)
        input_ = tf.concat([x1_input, x2_input], axis=-1)
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input_)
            output_ = model(input_)
        output_dx = tape.gradient(output_, input_)
        
        dx_dt = system.solve(input_)
        v_hat_dt = tf.reduce_sum((output_dx * dx_dt), axis=-1, keepdims=True)
        v_hat[i] = output_.numpy()[:, 0]
        v_dt_hat[i] = v_hat_dt.numpy()[:, 0]
        
    # ax = plt.axes(projection='3d')
    fig, axs = plt.subplots(ncols=2, figsize=(18, 8), subplot_kw={"projection":"3d"})
    axs[0].plot_wireframe(X1, X2, np.zeros_like(v_hat), color='gray', alpha=0.4)
    axs[0].plot_wireframe(X1, X2, v_hat, color='purple', alpha=0.4, label='V')
    axs[0].set_xlabel('X1')
    axs[0].set_ylabel('X2')
    
    axs[1].plot_wireframe(X1, X2, np.zeros_like(v_hat), color='gray', alpha=0.4)
    axs[1].plot_wireframe(X1, X2, v_dt_hat, color='purple', alpha=0.4, label='V_dot')
    axs[1].set_xlabel('X1')
    axs[1].set_ylabel('X2')
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=400)
    else:
        plt.show()

from system import Pendulum
from model import get_model
import os
import yaml

if __name__ == "__main__":
    pendulum = Pendulum()
    
    base_path = os.path.join('train_5')
    config_path = os.path.join(base_path, 'config.yaml')
    if not os.path.isfile(config_path):
        raise FileExistsError(config_path)
        
    with open(config_path) as f:
        hyperP = yaml.load(f, Loader=yaml.FullLoader)
        
    model = get_model(
        num_input=pendulum.num_x,
        num_output=1,
        num_layers=hyperP['num_layers'],
        num_nodes=hyperP['num_nodes'],
    )
    
    weights_path = os.path.join(base_path, 'model', 'variables', 'variables')
    model.load_weights(weights_path)
    
    draw_3d(model, pendulum)