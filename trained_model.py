import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

CLASSES = ['glioma', 'meningioma', 'pituitary', 'healthy']
IMG_SIZE = (224, 224)
MODEL_PATH = './xception_model_last.keras'
last_conv_layer_name = 'block14_sepconv2_act'


# Carregar o modelo completo uma vez
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()
print(f"TensorFlow version: {tf.__version__}")

# Refatorar a função para que ela separe as lógicas
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Acessa o modelo Xception que está dentro do modelo sequencial
    xception_model = model.get_layer("xception")
    
    # Cria o modelo para Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=xception_model.input,
        outputs=[xception_model.get_layer(last_conv_layer_name).output, xception_model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    return tf.keras.utils.array_to_img(superimposed_img)

def sample_images(data_dir='mri_images', classes=CLASSES, per_class=3):
    paths = []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(files)
        paths += files[:per_class]
    random.shuffle(paths)
    return paths

# Pega algumas imagens e exibe
# image_paths = sample_images('mri_images', CLASSES, per_class=3)[:9]
image_paths = ['mri_images/glioma/0000.jpg', 
               'mri_images/glioma/0031.jpg', 
               'mri_images/healthy/0044.jpg',
               'mri_images/healthy/1994.jpg',
               'mri_images/meningioma/1637.jpg',
               'mri_images/meningioma/1642.jpg',
               'mri_images/pituitary/0013.jpg',
               'mri_images/pituitary/1733.jpg',
               'mri_images/pituitary/1720.jpg']
plt.figure(figsize=(12,12))

for i, p in enumerate(image_paths):
    img = tf.keras.utils.load_img(p, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    preds = model.predict(arr, verbose=0)
    pred_class = np.argmax(preds[0])
    pred_label = CLASSES[pred_class]
    
    # Chama a função make_gradcam_heatmap
    heatmap = make_gradcam_heatmap(arr, model, last_conv_layer_name)
    sup_img = display_gradcam(p, heatmap)

    plt.subplot(3, 3, i+1)
    plt.imshow(sup_img)
    plt.title(f'Pred: {pred_label}')
    plt.axis('off')

plt.tight_layout()
plt.show()