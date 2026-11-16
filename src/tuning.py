"""KerasTuner search space for the churn MLP."""
import keras_tuner as kt

from src.nn_model import build_mlp


def build_hp_model(hp: kt.HyperParameters, input_dim: int):
    """Search space for KerasTuner. Each range is centred on the defaults
    used elsewhere in the project and bounded to values that keep training
    stable on a small (~7000-row) tabular dataset:
      - units_1/units_2: {32,64,128}/{16,32,64}, narrowing from block 1 to
        block 2 (standard funnel shape). Going above 128 units risked
        overfitting given the dataset size; below 16 units in block 2
        under-fit in early manual trials.
      - dropout: 0.1-0.5 in steps of 0.1 covers "light" to "heavy"
        regularisation; above 0.5 training became too unstable to
        converge within 20 epochs.
      - l2: {1e-5, 1e-4, 1e-3}, a log-spaced sweep from a very mild to a
        fairly strong weight penalty.
      - learning_rate: {1e-2, 1e-3, 5e-4}; 1e-2 was kept in the search
        mainly as an upper bound even though it was expected (and
        confirmed) to underperform, to verify the tuner correctly
        rejects it rather than silently never exploring that region.
    """
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
        # seeds the Oracle's own trial-sampling RNG so the same 10
        # hyperparameter combinations are tried on every run; this does not
        # fully remove randomness from each trial's TF training itself.
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
