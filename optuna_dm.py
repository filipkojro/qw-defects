from fktools import *
import tensorflow as tf
import sklearn
import optuna

def acc_from_hyperparams(trial: optuna.Trial):
    num_layers = trial.suggest_int(name="num_layers", low=0, high=6)

    layer_size_power = trial.suggest_int(name="layer_size_power", low=3, high=8)
    layer_size = 2**layer_size_power

    learning_rate = trial.suggest_float(name="learning_rate", )


layers = [
    tf.keras.layers.Input(shape=(8,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax'),
]

test_model = tf.keras.Sequential(layers)

test_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='kld',
    metrics=['acc']
)
# test_model.summary()


history = test_model.fit(
    X, y,
    epochs=100,
    batch_size=64,
    validation_split=0.2
)