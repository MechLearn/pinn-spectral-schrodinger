import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from .model import make_net
from .losses import compute_losses

def run_one_mode_learnE(n, save_dir="outputs"):

    os.makedirs(save_dir, exist_ok=True)

    USE_SINE = True if n >= 3 else False
    HIDDEN   = 128 if n >= 3 else 64
    N_col    = max(1024, 2048*n)
    EPOCHS   = 15000 if n >= 4 else (9000 if n==3 else (6000 if n==2 else 4000))
    LR0      = 3e-4  if n >= 4 else (5e-4 if n==3 else (7e-4 if n==2 else 1e-3))
    lam_hi, lam_lo = (300.0, 80.0) if n >= 3 else (40.0, 15.0 if n==2 else 10.0)

    net = make_net(n=n, hidden=HIDDEN, use_sine=USE_SINE)

    x_col = np.linspace(0, 1, N_col, dtype=np.float32).reshape(-1,1)
    x_batch = tf.constant(x_col)

    E_init = np.float32((n * math.pi)**2)
    alpha = tf.Variable(E_init, dtype=tf.float32)

    lr_sched = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=LR0,
        decay_steps=EPOCHS,
        end_learning_rate=LR0*0.1,
        power=1.0
    )

    opt = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0)

    vars_all = net.trainable_variables + [alpha]

    for ep in range(1, EPOCHS+1):

        lam = lam_hi if ep < EPOCHS//3 else lam_lo

        with tf.GradientTape() as tape:
            L, LPDE, Lnorm, integral, E = compute_losses(net, x_batch, alpha, lam)

        grads = tape.gradient(L, vars_all)
        opt.apply_gradients(zip(grads, vars_all))

        if ep % max(1000, EPOCHS//5) == 0 or ep == 1:
            print(f"n={n} ep={ep} E={E.numpy():.6f} L={L.numpy():.3e} ∫ψ²≈{integral.numpy():.6f}")

    return net, alpha
