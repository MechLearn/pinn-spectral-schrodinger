from pinn1d.config import load_config
from pinn1d.train import run_one_mode_learnE
import tensorflow as tf
import sys

if __name__ == "__main__":
    cfg = load_config("configs/base.yaml")

    # defaults
    n_min, n_max = 1, 20
    if len(sys.argv) >= 3:
        n_min = int(sys.argv[1])
        n_max = int(sys.argv[2])

    # semilla desde config
    tf.keras.utils.set_random_seed(int(cfg.get("seed", 0)))

    for n in range(n_min, n_max + 1):
        print(f"\n===== Running mode n={n} =====")
        run_one_mode_learnE(n, cfg)