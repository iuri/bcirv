# import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import matplotlib.pyplot as plt

import numpy as np

def create_cnn(input_shape):
    model = models.Sequential()
    # First convolutional layer with ReLU activation
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(3, 620, 1)))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Second convolutional layer
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Flatten the output for dense layers
    model.add(layers.Flatten())

    # Dense layer for feature learning
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output layer (for classification, modify the number of units accordingly)
    model.add(layers.Dense(2, activation='softmax'))  # Example: 2 classes

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Grad-CAM Function
def grad_cam(input_model, image, layer_name):
    grad_model = tf.keras.models.Model(inputs=input_model.inputs, outputs=[input_model.get_layer(layer_name).output, input_model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        class_id = np.argmax(predictions)
        class_loss = predictions[:, class_id]
    
    grads = tape.gradient(class_loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_output = conv_output[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(conv_output.shape[-1]):
        conv_output[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


def select_features_cnn1D(features, labels):
    # Input shape based on the data

    input_shape = features.shape[1], features.shape[2]  # (n_channels, n_timepoints)
    print('SHAPE', input_shape)
    cnn_model = create_cnn(input_shape)


    # Train the CNN model (assuming you have labels Y for classification)
    cnn_model.fit(features, labels, epochs=10, batch_size=32)
    

    # Get the heatmap
    heatmap = grad_cam(cnn_model, features, 'conv1d')  # 'conv1d' is the layer name for the first Conv1D layer

    # Display the heatmap
    plt.imshow(heatmap[0], cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.show()


def select_features_cnn2D(features, labels):

    print('shape', features.shape)
    ##
    # Select features with CNN 
    ## 
    # Normalize the combined features
    scaler = StandardScaler()
    features_flat = features.reshape(-1, features.shape[-1])
    features_flat = scaler.fit_transform(features_flat)
    features = features_flat.reshape(features.shape[0], features.shape[1], features.shape[2], 1)

    # Define the 2D CNN model
    def create_cnn_model(input_shape):
        model = Sequential([
            Input(shape=input_shape),
            
            # First Convolutional block
            Conv2D(32, (2, 2), activation='relu', padding='same'),
            MaxPooling2D((1, 2), padding='same'),
            
            # Second Convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((1, 3), padding='same'),

            # Third Convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((1, 2), padding='same'),
            # Flatten and fully connected layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')  # Output layer for 2 classes
        ])
        return model

    # Create and compile the model
    input_shape = (features.shape[1], features.shape[2], 1)  # (6, 623, 1)
    cnn_model = create_cnn_model(input_shape)
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Convert labels to one-hot encoding
    labels_one_hot = to_categorical(labels, num_classes=2)


    # Train the CNN model with validation split
    # history = cnn_model.fit(features, labels_one_hot, epochs=10, batch_size=32, validation_split=0.2)

    cnn_model.fit(features, labels_one_hot, epochs=10, batch_size=32, validation_split=0.2)

    # Extract features using the trained CNN model (from the second last layer)
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
    extracted_features = feature_extractor.predict(features)

    # print("Extracted Features Shape:", extracted_features.shape)  # Expected shape: (1357, 128)


    return extracted_features