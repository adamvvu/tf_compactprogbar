from tf_compactprogbar import CompactProgressBar
import pytest
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

np.random.seed(42)

def _GenerateData(N=1000, numFeatures=3):
    X = np.random.normal(size=(N,numFeatures))
    eps = np.random.normal(size=N, scale=0.1)
    Y = np.sum(np.log(np.abs(X)+1), axis=-1) + eps
    return X, Y

def _Compile_dnn(numFeatures, architecture=[32,32]):

    def my_metric(y_true, y_hat):
        return tf.math.abs(y_true-y_hat)
    
    def my_metric2(y_true, y_hat):
        return tf.math.square(y_true-y_hat)

    inputLayer = tfk.layers.Input(shape=(numFeatures,))
    x = inputLayer
    for nodes in architecture:
        x = tfk.layers.Dense(nodes, activation='relu')(x)
    x = tfk.layers.Dense(1)(x)

    model = tfk.Model(inputs=inputLayer, outputs=x)
    model.compile(optimizer=tfk.optimizers.Adam(), 
                  loss='mse', metrics=[my_metric, my_metric2])
    return model

def test_all(capsys):

    # Data
    X_train, Y_train = _GenerateData(100)
    X_test, Y_test = _GenerateData(20)

    # Compile model
    numFeatures = X_train.shape[-1]
    model = _Compile_dnn(numFeatures, architecture=[2])

    # Train
    EPOCHS = 3
    bar1 = CompactProgressBar(show_best=True, exclude=['my_metric2'])
    model.fit(X_train, Y_train,
                epochs=EPOCHS,
                batch_size=10,
                verbose=0,
                validation_data = (X_test, Y_test),
                callbacks=[bar1])
    
    ## Unit tests
    out = capsys.readouterr().err

    # Test: Progress bar descriptions
    assert "[Training]" in out
    assert f"/{EPOCHS}" in out

    # Test: Standard metrics
    assert "loss" in out
    assert "val_loss" in out
    assert "my_metric" in out
    assert "val_my_metric" in out

    # Test: Best/exclude
    assert "best_my_metric" in out
    assert "my_metric2" not in out

    return