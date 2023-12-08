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
        
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X1, X2, v_hat, color='gray', alpha=0.4, label='V')
    ax.plot_wireframe(X1, X2, v_dt_hat, color='purple', alpha=0.4, label='V_dot')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('')
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=400)
    else:
        plt.show()

from system import Pendulum
import os
if __name__ == "__main__":
    pendulum = Pendulum()
    load_model = tf.keras.models.load_model(os.path.join('train_5', 'model'))
    load_model.summary()
    
    # draw_3d(load_model, pendulum)