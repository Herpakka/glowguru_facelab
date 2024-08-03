import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os

# Load JSON data
with open('Jdataset.json') as f:
    data = json.load(f)

# Extract image paths, labels, and additional features
image_paths = [item['filename'] for item in data]
labels = [item['label'] for item in data]
features = [[item['sc_fore'], item['sc_foreP'], item['sc_cheek'], item['sc_foreD'], item['sc_foreS'], item['sc_decreaseP'], item['sc_denseP']] for item in data]

# Define a mapping for your labels
label_mapping = {'dry': 0, 'oily': 1, 'normal': 2}  # Update this mapping based on your specific labels
labels = np.array([label_mapping[label] for label in labels])

# Load and preprocess images
images = []
image_width = 500
image_height = 800
for path in image_paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (image_width, image_height))  # Adjust dimensions as needed
    images.append(img)
images = np.array(images)

# Normalize images
images = images.astype('float32') / 255.0

# Normalize additional features if needed
features = np.array(features).astype('float32')

# One-hot encode labels
labels = to_categorical(labels, num_classes=len(label_mapping))

# Split data into training and testing sets
X_train_images, X_test_images, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train_features, X_test_features = train_test_split(features, test_size=0.2, random_state=42)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate

# Image input branch
image_input = Input(shape=(image_height, image_width, 3))
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# Feature input branch
feature_input = Input(shape=(len(features[0]),))
y = Dense(64, activation='relu')(feature_input)

# Concatenate branches
combined = concatenate([x, y])
z = Dense(128, activation='relu')(combined)
z = Dropout(0.5)(z)
output = Dense(len(label_mapping), activation='softmax')(z)

# Create model
model = Model(inputs=[image_input, feature_input], outputs=output)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit([X_train_images, X_train_features], y_train, epochs=10, batch_size=32, validation_data=([X_test_images, X_test_features], y_test))

test_loss, test_acc = model.evaluate([X_test_images, X_test_features], y_test)
print(f'Test accuracy: {test_acc}')