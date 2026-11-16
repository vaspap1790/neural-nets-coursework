"""Feedforward neural network (MLP) for churn prediction, with a residual
block, batch normalization, dropout and L1/L2 regularisation."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def residual_block(x, units: int, dropout: float, l2: float):
    """Dense -> BN -> ReLU -> Dropout -> Dense -> BN, added back to a
    linearly-projected shortcut of the block's input, then ReLU. This lets
    the network learn an identity mapping by default and keeps gradients
    flowing to earlier layers, the same motivation as in image-domain
    residual networks, adapted here to a plain tabular MLP."""
    shortcut = layers.Dense(units, kernel_regularizer=regularizers.l2(l2))(x)

    h = layers.Dense(units, kernel_regularizer=regularizers.l2(l2))(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("relu")(h)
    h = layers.Dropout(dropout)(h)
    h = layers.Dense(units, kernel_regularizer=regularizers.l2(l2))(h)
    h = layers.BatchNormalization()(h)

    out = layers.Add()([shortcut, h])
    return layers.Activation("relu")(out)


def build_mlp(
    input_dim: int,
    units_1: int = 64,
    units_2: int = 32,
    dropout: float = 0.3,
    l2: float = 1e-4,
    l1_output: float = 1e-5,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Two residual blocks (units_1 then units_2, narrowing as is typical
    for a tabular MLP) followed by a single sigmoid output unit.

    Regularisation is deliberately layered rather than relying on one
    technique: L2 inside every Dense layer discourages large weights
    without zeroing them out (keeps every input feature "available"),
    Dropout inside each block randomly deactivates units during training
    so the network cannot over-rely on any single neuron, BatchNorm
    stabilises the distribution of layer inputs (letting a higher
    learning rate be used safely), and a small L1 penalty on the output
    layer encourages sparsity in how the final decision combines the
    two blocks' outputs. None of these alone fully prevents
    overfitting on a dataset this size, but combined with the
    EarlyStopping callback used at training time (see
    scripts/q3_neural_network.py) they keep the train/validation loss
    gap small (Figure 7 in the report).
    """
    inputs = keras.Input(shape=(input_dim,))
    x = residual_block(inputs, units_1, dropout, l2)
    x = residual_block(x, units_2, dropout, l2)
    outputs = layers.Dense(
        1, activation="sigmoid", kernel_regularizer=regularizers.l1(l1_output)
    )(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        # AUC is threshold-independent and robust to class imbalance, so it
        # is tracked (and used as the tuning objective) alongside accuracy
        # rather than relying on accuracy alone, which a majority-class
        # predictor could inflate on this imbalanced dataset.
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


def class_weights(y_train) -> dict:
    """Mirrors class_weight='balanced' from the Logistic Regression
    baseline: inversely proportional to class frequency, so the minority
    (Churn=1) class is not drowned out during training."""
    y = np.asarray(y_train)
    n = len(y)
    n_pos = y.sum()
    n_neg = n - n_pos
    return {0: n / (2 * n_neg), 1: n / (2 * n_pos)}
