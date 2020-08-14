import tensorflow.keras as keras
from classification_models.tfkeras import Classifiers


def simple_model(input_shape, n_classes):
    inputs = keras.layers.Input(shape=input_shape, name='digits')
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(64, activation='relu', name='dense_1')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = keras.layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def simple_cnn_model(input_shape, n_classes):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(n_classes))
    model.add(keras.layers.Activation('softmax'))

    return model


def create_keras_classification_model(source, architecture, input_shape, n_classes, pretrained=False):
    assert input_shape[-1] in [1, 3], 'The input shape is incompatible with the model.'
    if source.startswith('cm'):
        # Create the model using the classification_models repository
        Architecture, preprocessing = Classifiers.get(architecture)
        weights = 'imagenet' if pretrained else None
        model = Architecture(input_shape=input_shape, classes=n_classes, weights=weights, include_top=not pretrained)

        if pretrained:
            # Perform model surgery and add an output softmax layer
            new_output = keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
            new_output = keras.layers.Dense(n_classes)(new_output)
            if source == 'cm_cxr':
                # Models that do multi-label classification use sigmoid outputs
                new_output = keras.activations.sigmoid(new_output)
            else:
                # Standard softmax output is best for most cases
                new_output = keras.activations.softmax(new_output)
            model = keras.Model(inputs=model.inputs, outputs=new_output)

    elif source == 'simple_cnn':
        model = simple_cnn_model(input_shape, n_classes)
    else:
        raise NotImplementedError
    # Print the model summary
    print(model.summary())
    return model


def freeze_all_layers_except_last_linear_layer(model):
    """
    Freezes all the layers in a model except the last Dense layer.
    According to https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization,
    setting trainable = False for BatchNorm:
    - freezes the weights
    - runs the layer in inference mode
    """
    # Set all layers to be not trainable
    for layer in model.layers:
        layer.trainable = False

    # Find the last linear layer
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Dense):
            layer.trainable = True
            break


def reinitialize_last_linear_layer(model):
    # Loop over the layers in reverse
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Dense):
            # Compute the shapes for this layer
            kernel_shape = [dim.value for dim in layer.kernel.shape.dims]
            bias_shape = [dim.value for dim in layer.bias.shape.dims]

            # Initialize using Glorot Uniform for the weights, and zeros for the biases
            init_kernel_weights = keras.initializers.glorot_uniform()(kernel_shape)
            init_bias_weights = keras.initializers.zeros()(bias_shape)
            layer.set_weights([init_kernel_weights, init_bias_weights])

            break
