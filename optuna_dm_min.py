import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fktools import *
import tensorflow as tf
import sklearn
import optuna
from dist_metric import DistributionOverlap

# loading data with same split every time
X = np.load("dataset_denoising_multiple_X.npz")['arr_0']
y = np.load("dataset_denoising_multiple_y.npz")['arr_0']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

def loss_from_hyperparams(trial: optuna.Trial):

    # trying numbers using optuna
    num_layers = trial.suggest_int(name="num_layers", low=1, high=6)

    layer_size_power = trial.suggest_int(name="layer_size_power", low=3, high=8)
    layer_size = 2**layer_size_power

    learning_rate = trial.suggest_float(name="learning_rate", low=1e-5, high=1e-2, log=True)

    batch_size_power = trial.suggest_int(name="batch_size_power", low=4, high=8)
    batch_size = 2**batch_size_power

    epochs = trial.suggest_int(name="epochs", low=8, high=128)

    # creating model
    layers = [tf.keras.layers.Input(shape=(8,))]

    for i in range(num_layers):
        layers.append(tf.keras.layers.Dense(layer_size, activation='relu'))

    layers.append(tf.keras.layers.Dense(8, activation='softmax'))

    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = tf.keras.losses.KLDivergence(),
        metrics=[DistributionOverlap()]
    )

    # model learning
    model.fit(
        X_train,
        y_train,
        verbose = 0,
        epochs = epochs,
        batch_size = batch_size
    )

    loss, dist_overlap = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    num_params = model.count_params()

    return dist_overlap, num_params


# running hyperparameter optimization
study = optuna.create_study(
    directions=["maximize", "minimize"],
    storage="sqlite:///db.sqlite3",
    study_name="denoising_dense_min",
    load_if_exists=True,
)
study.optimize(loss_from_hyperparams, n_trials=None, n_jobs=6, show_progress_bar=True)

print(f"BEST PARAMS: {study.best_params}")