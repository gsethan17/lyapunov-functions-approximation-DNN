import tensorflow as tf

class RollLoss(tf.keras.losses.Loss):
    """
    Helmholtz Loss for physics-informed neural network.

    Parameters
    ----------
    pde: HelmholtzPDE
        The HelmholtzPDE object representing the PDE
        The name of the loss, by default 'ReLoBRaLoHelmholtzLoss'
    """
    def __init__(self, name='RollLoss'):
        super().__init__(name=name)

    def __call__(self, losses):
        losses = [tf.reduce_mean(loss) for loss in losses]
        return tf.reduce_mean(losses)
    
class ReLoBRaLoRollLoss(RollLoss):
    """
    Class for the ReLoBRaLo Helmholtz Loss.
    This class extends the Helmholtz Loss to have dynamic weighting for each term in the calculation of the loss.
    """
    def __init__(self, num_terms, alpha:float=0.999, temperature:float=1., rho:float=0.9999,
                 name='ReLoBRaLoRollLoss'):
        """
        Parameters
        ----------
        pde : HelmholtzPDE
            An instance of HelmholtzPDE class containing the `compute_loss` function.
        alpha, optional : float
            Controls the exponential weight decay rate.
            Value between 0 and 1. The smaller, the more stochasticity.
            0 means no historical information is transmitted to the next iteration.
            1 means only first calculation is retained. Defaults to 0.999.
        temperature, optional : float
            Softmax temperature coefficient. Controlls the "sharpness" of the softmax operation.
            Defaults to 1.
        rho, optional : float
            Probability of the Bernoulli random variable controlling the frequency of random lookbacks.
            Value berween 0 and 1. The smaller, the fewer lookbacks happen.
            0 means lambdas are always calculated w.r.t. the initial loss values.
            1 means lambdas are always calculated w.r.t. the loss values in the previous training iteration.
            Defaults to 0.9999.
        """
        super().__init__(name=name)
        self.num_terms = num_terms
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        self.call_count = tf.Variable(0, trainable=False, dtype=tf.int16)

        self.lambdas = [tf.Variable(1., trainable=False) for _ in range(self.num_terms)]
        self.last_losses = [tf.Variable(1., trainable=False) for _ in range(self.num_terms)]
        self.init_losses = [tf.Variable(1., trainable=False) for _ in range(self.num_terms)]

        self.eps = 1e-7

    def __call__(self, losses):
        if len(losses) != self.num_terms:
            raise ValueError("num_terms and losses must have the same length")
        # x, t = xt[:, :1], xt[:, 1:]
        losses = [tf.reduce_mean(loss) for loss in losses]

        # in first iteration (self.call_count == 0), drop lambda_hat and use init lambdas, i.e. lambda = 1
        #   i.e. alpha = 1 and rho = 1
        # in second iteration (self.call_count == 1), drop init lambdas and use only lambda_hat
        #   i.e. alpha = 0 and rho = 1
        # afterwards, default procedure (see paper)
        #   i.e. alpha = self.alpha and rho = Bernoully random variable with p = self.rho
        alpha = tf.cond(tf.equal(self.call_count, 0),
                lambda: 1.,
                lambda: tf.cond(tf.equal(self.call_count, 1),
                                lambda: 0.,
                                lambda: self.alpha))
        rho = tf.cond(tf.equal(self.call_count, 0),
              lambda: 1.,
              lambda: tf.cond(tf.equal(self.call_count, 1),
                              lambda: 1.,
                              lambda: tf.cast(tf.random.uniform(shape=()) < self.rho, dtype=tf.float32)))

        # compute new lambdas w.r.t. the losses in the previous iteration
        lambdas_hat = [losses[i] / (self.last_losses[i] * self.temperature + self.eps) for i in range(len(losses))]
        lambdas_hat = tf.nn.softmax(lambdas_hat - tf.reduce_max(lambdas_hat)) * tf.cast(len(losses), dtype=tf.float32)

        # compute new lambdas w.r.t. the losses in the first iteration
        init_lambdas_hat = [losses[i] / (self.init_losses[i] * self.temperature + self.eps) for i in range(len(losses))]
        init_lambdas_hat = tf.nn.softmax(init_lambdas_hat - tf.reduce_max(init_lambdas_hat)) * tf.cast(len(losses), dtype=tf.float32)

        # use rho for deciding, whether a random lookback should be performed
        new_lambdas = [(rho * alpha * self.lambdas[i] + (1 - rho) * alpha * init_lambdas_hat[i] + (1 - alpha) * lambdas_hat[i]) for i in range(len(losses))]
        self.lambdas = [var.assign(tf.stop_gradient(lam)) for var, lam in zip(self.lambdas, new_lambdas)]

        # compute weighted loss
        loss = tf.reduce_sum([lam * loss for lam, loss in zip(self.lambdas, losses)])

        # store current losses in self.last_losses to be accessed in the next iteration
        self.last_losses = [var.assign(tf.stop_gradient(loss)) for var, loss in zip(self.last_losses, losses)]
        # in first iteration, store losses in self.init_losses to be accessed in next iterations
        first_iteration = tf.cast(self.call_count < 1, dtype=tf.float32)
        self.init_losses = [var.assign(tf.stop_gradient(loss * first_iteration + var * (1 - first_iteration))) for var, loss in zip(self.init_losses, losses)]

        self.call_count.assign_add(1)

        return loss
    
from collections import defaultdict

class LossTracking:
    def __init__(self):
        self.mean_total_loss = tf.keras.metrics.Mean()
        self.mean_IC0_loss = tf.keras.metrics.Mean()
        self.mean_IC1_loss = tf.keras.metrics.Mean()
        self.mean_PDE_loss = tf.keras.metrics.Mean()
        self.loss_history = defaultdict(list)
        self.num_terms = 3

    def update(self, total_loss, IC0_loss, IC1_loss, PDE_loss):
        self.mean_total_loss(total_loss)
        self.mean_IC0_loss(IC0_loss)
        self.mean_IC1_loss(IC1_loss)
        self.mean_PDE_loss(PDE_loss)

    def reset(self):
        self.mean_total_loss.reset_states()
        self.mean_IC0_loss.reset_states()
        self.mean_IC1_loss.reset_states()
        self.mean_PDE_loss.reset_states()

    def print(self):
        print(f"IC0={self.mean_IC0_loss.result().numpy():.4e}, \
                IC1={self.mean_IC1_loss.result().numpy():.4e}, \
                PDE={self.mean_PDE_loss.result().numpy():.4e}, \
                total_loss={self.mean_total_loss.result().numpy():.4e}")
        
    def history(self):
        self.loss_history['total_loss'].append(self.mean_total_loss.result().numpy())
        self.loss_history['IC0_loss'].append(self.mean_IC_loss.result().numpy())
        self.loss_history['IC1_loss'].append(self.mean_IC_loss.result().numpy())
        self.loss_history['PDE_loss'].append(self.mean_ODE_loss.result().numpy())