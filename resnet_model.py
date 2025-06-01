import os
import cv2
import numpy as np
import pandas as pd
from keras_resnet.models import ResNet50
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Input
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

input_tensor = Input(shape=(224, 224, 3), batch_size=32)


# csv file
df = pd.read_csv(r'data\dress_patterns.csv')

# for img paths
image_dir = r'C:\Users\kvjai\ML PROJECTS\Clothing items classification\data\dataset_category'  
id_to_path = {}

# map
for category in os.listdir(image_dir):
    category_path = os.path.join(image_dir, category)
    if os.path.isdir(category_path):
        for filename in os.listdir(category_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                _unit_id = os.path.splitext(filename)[0]
                image_path = os.path.join(category_path, filename)
                id_to_path[_unit_id] = (image_path, category) # path and label


input_shape = (224, 224, 3)
print(f"\n\nInput shape: {input_tensor}")


# pretrained ResNet50 for Feature Extraction

base_model = ResNet50(weights='imagenet', include_top=False, inputs=input_tensor, freeze_bn=True)
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

# image processing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to match ResNet50 input
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rgb
    img = img / 255.0  
    return img

# extract features and labels
features = []
labels = []

for index, row in df.iterrows():
    _unit_id = str(row['_unit_id'])  #
    if _unit_id in id_to_path:
        image_path, category = id_to_path[_unit_id]
        image = preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        feature = feature_extractor.predict(image)
        features.append(feature.flatten())
        labels.append(category)


features = np.array(features)
labels = np.array(labels)

# label encoding
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

# SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)

# evaluate
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


def predict_image(image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)
    feature = feature_extractor.predict(image).flatten().reshape(1, -1)
    prediction = svm_classifier.predict(feature)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

new_image_path = r'C:\Users\kvjai\ML PROJECTS\Clothing items classification\data\dataset_category\floral\851505511.jpg'
predicted_category = predict_image(new_image_path)
print(f'Predicted Category: {predicted_category}')
