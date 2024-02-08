from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model

import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_features(preprocessed_images):
    # Load the MobileNetV2 model, pretrained on ImageNet, without the top layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Predict to extract features
    features = model.predict(preprocessed_images)

    # Flatten the features to fit the clustering algorithm
    features_flattened = features.reshape(features.shape[0], -1)

    return features_flattened



def cluster_images(features, num_clusters=8):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=num_clusters, random_state=None)
    kmeans.fit(features_scaled)
    return kmeans.labels_
