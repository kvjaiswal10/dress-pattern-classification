import os
import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils  import to_categorical
#from keras.src.legacy.preprocessing.image import img_to_array, load_img
from keras_preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

from tf_keras.applications import MobileNetV2

from tf_keras.models import Sequential, Model
from tf_keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping
# ------------------------ 1️⃣ Load CSV and Map Labels ------------------------

# Load CSV file
csv_path = r'C:\Users\kvjai\ML PROJECTS\Clothing items classification\data\dress_patterns.csv'
df = pd.read_csv(csv_path)

# Define dataset path
dataset_path = r'C:\Users\kvjai\ML PROJECTS\Clothing items classification\data\dataset_category'

# Convert labels to numerical values
label_map = {label: idx for idx, label in enumerate(df["category"].unique())}
df["label"] = df["category"].map(label_map)

# ------------------------ 2️⃣ Load Images and Preprocess ------------------------

img_size = (224, 224)  # Image size
images = []
labels = []

missing_images = 0  # Track missing images

for _, row in df.iterrows():
    img_name = str(row["_unit_id"]) + ".jpg"
    img_found = False
    
    for subfolder in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, subfolder, img_name)
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
            labels.append(row["label"])
            img_found = True
            break  # Stop searching once found

    if not img_found:
        missing_images += 1

print(f"\n\nTotal images loaded: {len(images)}")
print(f"Missing images: {missing_images}\n\n")


# Convert lists to NumPy arrays
images = np.array(images)
labels = to_categorical(labels, num_classes=len(label_map))  # One-hot encode labels

print(f"\n\nTotal images loaded: {len(images)}\n\n")

# ------------------------ 3️⃣ Train-Test Split ------------------------

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.3, height_shift_range=0.3,
    shear_range=0.3, zoom_range=0.3, horizontal_flip=True
)


# ------------------------ 4️⃣ Define CNN Model ------------------------

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(len(label_map), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=x)
# ------------------------ 5️⃣ Compile & Train Model ------------------------

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

train_generator = datagen.flow(X_train, y_train, batch_size=32)
val_generator = datagen.flow(X_val, y_val, batch_size=32)

early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    steps_per_epoch=len(X_train) // 32,
    validation_steps=len(X_val) // 32,
    callbacks=[early_stopping]
)


# ------------------------ 6️⃣ Save & Evaluate Model ------------------------

model.save("dress_pattern_classifier.h5")
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
