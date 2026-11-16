"""KerasTuner search space for the churn MLP."""
import keras_tuner as kt

from src.nn_model import build_mlp


def build_hp_model(hp: kt.HyperParameters, input_dim: int):
    return build_mlp(
        input_dim=input_dim,
        units_1=hp.Choice("units_1", [32, 64, 128]),
        units_2=hp.Choice("units_2", [16, 32, 64]),
        dropout=hp.Float("dropout", 0.1, 0.5, step=0.1),
        l2=hp.Choice("l2", [1e-5, 1e-4, 1e-3]),
        learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 5e-4]),
    )


def run_search(
    X_train,
    y_train,
    input_dim: int,
    max_trials: int = 10,
    epochs: int = 20,
    directory: str = "outputs/kt_search",
):
    tuner = kt.RandomSearch(
        lambda hp: build_hp_model(hp, input_dim),
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=max_trials,
        executions_per_trial=1,
        seed=42,
        directory=directory,
        project_name="churn_mlp",
        overwrite=True,
    )
    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        verbose=0,
    )
    best_hp = tuner.get_best_hyperparameters(1)[0]
    return tuner, best_hp
