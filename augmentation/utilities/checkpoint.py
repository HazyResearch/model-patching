import pickle
import gzip


def compile_keras_models(models, optimizers):
    # Compile the models: this is necessary in order to save model architecture, weights and optimizer to disk
    # It doesn't matter what loss we use here since we're not going to be calling model.fit: TODO check!
    for model, optimizer in zip(models, optimizers):
        model.compile(optimizer=optimizer, loss='mse')
        # Calling _make_train_function populates the optimizer with per-variable weights
        model._make_train_function()


def save_tf_optimizer_state(optimizer, store_path, zip=True):
    if zip:
        open = gzip.open

    with open(store_path, 'wb') as f:
        pickle.dump(optimizer.get_weights(), f)


def load_tf_optimizer_state(optimizer, load_path, zip=True):
    if zip:
        open = gzip.open

    with open(load_path, 'rb') as f:
        optimizer.set_weights(pickle.load(f))

