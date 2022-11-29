import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def get_class_names(json_path,labels):
    with open(json_path, 'r') as f:
        flower_classes = json.load(f)
    classes = list(map(lambda y: flower_classes[str(y)],labels.numpy().squeeze()))
    return classes


def process_images(image_path):
    
    im = Image.open(image_path)
    test_image = np.asarray(im)
    image = tf.cast(test_image, tf.float32)
    resized_image = tf.image.resize(image, (224, 224))
    resized_image /= 255
    return resized_image.numpy()

def predict (image_path, model, top_k):
    
    processed_image = process_images(image_path)
    final_image = np.expand_dims(processed_image, axis = 0)
    model = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
    result = model.predict(final_image)
    
    top_k_probs, top_k_labels = tf.nn.top_k(result, k=top_k)

    return top_k_probs, top_k_labels