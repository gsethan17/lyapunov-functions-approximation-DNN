import tensorflow as tf
epsilon = 1e-4

@tf.function
def feed_forward(x, model):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y_hat = model(x)
    y_dx = tape.gradient(y_hat, x)
    
    return y_hat, y_dx
    
@tf.function
def get_zero_residual(x, model, system):
    v_hat, v_hat_dx = feed_forward(x, model)
    dx_dt = system.solve(x)
    v_hat_dt = tf.reduce_sum((v_hat_dx * dx_dt), axis=-1, keepdims=True)
    
    loss_v_zero = tf.reduce_mean(tf.square(v_hat))
    loss_v_dt_zero = tf.reduce_mean(tf.square(v_hat_dt))
    # print(v_hat_dt)
    # print(v_hat_dt, loss_v_dt_zero)
    return [loss_v_zero, loss_v_dt_zero]
    
@tf.function
def get_residual(x, model, system):
    v_hat, v_hat_dx = feed_forward(x, model)
    dx_dt = system.solve(x)
    v_hat_dt = tf.reduce_sum((v_hat_dx * dx_dt), axis=-1, keepdims=True)
    
    v_mask = tf.less_equal(v_hat, tf.zeros_like(v_hat))
    v_dt_mask = tf.greater_equal(v_hat_dt, tf.zeros_like(v_hat_dt))
    
    v_hat_loss = tf.boolean_mask(tf.square(v_hat) + epsilon, v_mask)
    v_hat_dt_loss = tf.boolean_mask(tf.square(v_hat_dt) + epsilon, v_dt_mask)
    
    loss_v = tf.reduce_mean(v_hat_loss)
    loss_v_dt = tf.reduce_mean(v_hat_dt_loss)
    
    return [loss_v, loss_v_dt]
    
    
@tf.function
def train_step(x, x_zero, model, system):
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_weights)
        losses_zero = get_zero_residual(x_zero, model, system)
        losses_nonZero = get_residual(x, model, system)
    
        losses = losses_zero + losses_nonZero
    
        # Weight loss
        total_loss = model.loss(losses)

    # gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    return losses, total_loss, gradients

@tf.function
def val_step(x, x_zero, model, system):
    losses_zero = get_zero_residual(x_zero, model, system)
    losses_nonZero = get_residual(x, model, system)
    
    losses = losses_zero + losses_nonZero
    
    return losses