from system import Pendulum
from model import get_model
from dataloader import DataLoader
from train import train_step, val_step
from loss import ReLoBRaLoRollLoss
from plot import draw_3d
import os
from glob import glob
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, CallbackList
import numpy as np
import yaml

def main(hyperP):
    log_path = os.path.join(os.path.dirname(__file__), 'runs', 'train')
    if os.path.isdir(log_path):
        num_path = len(glob(log_path+'*'))
        log_path += f'_{num_path}'
    os.makedirs(os.path.join(log_path, 'img'))
    os.makedirs(os.path.join(log_path, 'model'))
            
    # save configuration
    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        yaml.dump(hyperP, f)
            
    pendulum = Pendulum()
    
    V = get_model(
        num_input=pendulum.num_x,
        num_output=1,
        num_layers=hyperP['num_layers'],
        num_nodes=hyperP['num_nodes'],
        use_bias=hyperP['use_bias'],
        use_last_bias=hyperP['use_last_bias'],
    )
    
    V.summary()
    # set loss
    loss_f = ReLoBRaLoRollLoss(num_terms=4)
    
    # Set up optimizer
    optimizer = Adam(learning_rate=1e-3)
    
    V.compile(loss=loss_f,
                optimizer=optimizer)
    
    dataloader = DataLoader(
        system=pendulum,
        batch_size=hyperP['batch_size'],
        batch_size_zero=hyperP['batch_size_zero'],
    )
    
    _callbacks = [
                TensorBoard(
                    log_dir=os.path.join(log_path, 'logs'),
                    update_freq='epoch',
                    ),
                # EarlyStopping(
                    # monitor='metric', 
                    # patience=1000, 
                    # restore_best_weights=False, 
                    # verbose=1,
                    # mode='min',
                    # ),
    ]
    callbacks = CallbackList(
                    _callbacks, add_history=False, model=V)
    
    callbacks.on_train_begin()
    for epoch in range(hyperP['num_epoch']):
        train_losses = {
            'V_zero':[],
            'V_dt_zero':[],
            'V':[],
            'V_dt':[],
        }
        val_losses = {
            'V_zero':[],
            'V_dt_zero':[],
            'V':[],
            'V_dt':[],
        }
        for step in range(hyperP['step_per_epoch']):
            x, x_zero = dataloader()
            losses, total_loss, gradients = train_step(x, x_zero, V, pendulum)
            V.optimizer.apply_gradients(zip(gradients, V.trainable_variables))
            
            train_losses['V_zero'].append(losses[0].numpy())
            train_losses['V_dt_zero'].append(losses[1].numpy())
            train_losses['V'].append(losses[2].numpy())
            train_losses['V_dt'].append(losses[3].numpy())
            
        # validation step
        X1 = np.linspace(pendulum.range_x['x1']['min'], pendulum.range_x['x1']['max'], 100)
        X2 = np.linspace(pendulum.range_x['x2']['min'], pendulum.range_x['x2']['max'], 100)
        
        # generate validation x
        val_x = np.array([])
        for x1 in X1:
            for x2 in X2:
                if len(val_x) == 0:
                    val_x = np.array([[x1, x2]])
                else:
                    val_x = np.concatenate((val_x, np.array([[x1, x2]])), axis=0)
        if np.prod(np.invert(np.isclose(val_x, 0.0))) == 0:
            raise ValueError("validation x has zero elements")
        
        val_x_zero = np.zeros((1, 2))
        val_x = tf.convert_to_tensor(val_x, dtype=tf.float32)
        val_x_zero = tf.convert_to_tensor(val_x_zero, dtype=tf.float32)
        
        losses = val_step(val_x, val_x_zero, V, pendulum)
        val_losses['V_zero'].append(losses[0].numpy())
        val_losses['V_dt_zero'].append(losses[1].numpy())
        val_losses['V'].append(losses[2].numpy())
        val_losses['V_dt'].append(losses[3].numpy())
        
        logs = {
            'V_zero':np.mean(train_losses['V_zero']),
            'V_dt_zero':np.mean(train_losses['V_dt_zero']),
            'V':np.mean(train_losses['V']),
            'V_dt':np.mean(train_losses['V_dt']),
            'val_V_zero':np.mean(val_losses['V_zero']),
            'val_V_dt_zero':np.mean(val_losses['V_dt_zero']),
            'val_V':np.mean(val_losses['V']),
            'val_V_dt':np.mean(val_losses['V_dt']),
        }
        
        callbacks.on_epoch_end(epoch, 
                            logs=logs,
                            )
        
        flag1 = np.isclose(logs['val_V_zero'], 0.0, atol=1e-7)
        flag2 = np.isclose(logs['val_V_dt_zero'], 0.0, atol=1e-7)
        flag3 = np.isclose(logs['val_V'], 0.0, atol=1e-7)
        flag4 = np.isclose(logs['val_V_dt'], 0.0, atol=1e-7)
        
        if flag1 and flag2 and flag3 and flag4:
            break
        
        if epoch % 10 == 0:
            draw_3d(V, pendulum, save_path=os.path.join(log_path, 'img', f'{epoch}.png'))
        print(f'{epoch}: {flag1}, {flag2}, {flag3}, {flag4}')
        print(losses)
            
    draw_3d(V, pendulum, save_path=os.path.join(log_path, 'img', f'{epoch}.png'))
    V.save(os.path.join(log_path, 'model'))


from utils import gpu_limit
if __name__ == "__main__":
    gpu_limit(1)
    
    hyperP = {
        'num_layers':6,
        'num_nodes':64,
        'batch_size':512,
        'batch_size_zero':64,
        'step_per_epoch':100,
        # 'step_per_epoch':0,
        'num_epoch':1000,
        'use_bias':True,
        'use_last_bias':False,
    }
    
    main(hyperP)
    # draw()
