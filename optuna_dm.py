from fktools import *
import tensorflow as tf
import sklearn
import optuna

# loading data with same split every time
X = np.load("dataset_denoising_multiple_X.npz")['arr_0']
y = np.load("dataset_denoising_multiple_y.npz")['arr_0']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

def loss_from_hyperparams(trial: optuna.Trial):

    # trying numbers using optuna
    num_layers = trial.suggest_int(name="num_layers", low=0, high=6)

    layer_size_power = trial.suggest_int(name="layer_size_power", low=3, high=8)
    layer_size = 2**layer_size_power

    learning_rate = trial.suggest_float(name="learning_rate", low=0.00001, high=0.01)

    batch_size_power = trial.suggest_int(name="batch_size_power", low=4, high=8)
    batch_size = 2**batch_size_power

    epochs = trial.suggest_int(name="epochs", low=10, high=100)

    batch_size = 1024


    # creating model
    layers = [tf.keras.layers.Input(shape=(8,))]

    for i in range(num_layers):
        layers.append(tf.keras.layers.Dense(layer_size, activation='relu'))

    layers.append(tf.keras.layers.Dense(8, activation='softmax'))

    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = tf.keras.losses.KLDivergence(),
    )

    # model learning
    model.fit(
        X_train,
        y_train,
        verbose = 0,
        epochs = epochs,
        batch_size = batch_size
    )

    loss = model.evaluate(X_test, y_test)

    return loss


# setting to run on GPU
# if len(tf.config.list_logical_devices('GPU')) > 0:
#     tf.device(tf.config.list_logical_devices('GPU')[0].name)
tf.device(tf.config.list_logical_devices('CPU')[0].name)

# running hyperparameter optimization
study = optuna.create_study(
    direction="minimize"
)
study.optimize(loss_from_hyperparams, n_trials=10, show_progress_bar=True)

print(f"BEST PARAMS: {study.best_params}")