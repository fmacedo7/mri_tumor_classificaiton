import os
import time
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import visualkeras
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import categorical_crossentropy # type: ignore
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint # type: ignore
from tensorflow.keras.regularizers import l1, l2 # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

BATCH_SIZE = 32
IMAGE_SIZE = 224
SEED = 123
DATA_DIR = './mri_images'
CLASSES = ['glioma', 'meningioma', 'pituitary', 'healthy']
all_filepaths = []
all_labels =[]

class_to_index = {name: i for i, name in enumerate(CLASSES)}
for class_name in CLASSES:
    class_dir = os.path.join(DATA_DIR, class_name)
    for filename in os.listdir(class_dir):
        filepath = os.path.join(class_dir, filename)
        all_filepaths.append(filepath)
        all_labels.append(class_to_index[class_name])

# Converter para arrays numpy para usar com train_test_split
all_filepaths = np.array(all_filepaths)
all_labels = np.array(all_labels)

# Primeiro split: Treino (80%) e Temp (20%)
train_filepaths, temp_filepaths, train_labels, temp_labels = train_test_split(
    all_filepaths, all_labels,
    test_size=0.2, # 20% para temp (validação + teste)
    random_state=SEED,
    stratify=all_labels
)

# Segundo split: Validação (50% de Temp, i.e., 10% do total) e Teste (50% de Temp, i.e., 10% do total)
val_filepaths, test_filepaths, val_labels, test_labels = train_test_split(
    temp_filepaths, temp_labels,
    test_size=0.5, # 50% dos 20% temporários = 10% do total
    random_state=SEED,
    stratify=temp_labels
)

print(f"Número de imagens de Treinamento: {len(train_filepaths)}")
print(f"Número de imagens de Validação: {len(val_filepaths)}")
print(f"Número de imagens de Teste: {len(test_filepaths)}")


def preprocess_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0 # Normalização
    # Convert the label to one-hot encoding
    label = tf.one_hot(label, depth=len(CLASSES))
    return image, label

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(factor=0.15),
])

def prepare_tf_dataset(filepaths, labels, shuffle_buffer_size=None, augment=False):
    # Use tf.constant para garantir que os arrays NumPy sejam convertidos em tensores TensorFlow
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(filepaths), tf.constant(labels)))

    # Mapear a função de pré-processamento
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Aplicar augmentation APENAS para o conjunto de treinamento
    if augment:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        # Embaralhar o conjunto de treinamento APÓS o augmentation (para melhor aleatoriedade)
        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=SEED)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Criar os datasets de treinamento, validação e teste
train_ds = prepare_tf_dataset(train_filepaths, train_labels, shuffle_buffer_size=1000, augment=True) # Buffersize pode ser len(train_filepaths) se a memória permitir
val_ds = prepare_tf_dataset(val_filepaths, val_labels, shuffle_buffer_size=None, augment=False)
test_ds = prepare_tf_dataset(test_filepaths, test_labels, shuffle_buffer_size=None, augment=False)

val_ds = val_ds.cache()
test_ds = test_ds.cache()

class_names = CLASSES
print("Class names:", class_names)

def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Verificando se val_loss e val_accuracy não estão vazios antes de usar index()
    if val_loss and val_accuracy:
        best_epoch = val_loss.index(min(val_loss)) + 1
        best_accuracy_epoch = val_accuracy.index(max(val_accuracy)) + 1

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(loss) + 1), loss, label='Treinamento')
        plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validação')
        plt.scatter(best_epoch, min(val_loss), color='red', label=f'Melhor Época ({best_epoch})', zorder=5)
        plt.annotate(f"{min(val_loss):.4f}", (best_epoch, min(val_loss)), textcoords="offset points", xytext=(0,10), ha='center', color='red')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()
        plt.title('Histórico de Perda')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(accuracy) + 1), accuracy, label='Treinamento')
        plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validação')
        plt.scatter(best_accuracy_epoch, max(val_accuracy), color='green', label=f'Melhor Época ({best_accuracy_epoch})', zorder=5)
        plt.annotate(f"{max(val_accuracy):.4f}", (best_accuracy_epoch, max(val_accuracy)), textcoords="offset points", xytext=(0,10), ha='center', color='green')
        plt.xlabel('Época')
        plt.ylabel('Precisão')
        plt.legend()
        plt.title('Histórico de Precisão')

        plt.tight_layout()
        plt.show()
    else:
        print("Erro: Os dados de validação não estão presentes no histórico.")

try:
    batch_size = 32
    print("Data generators criados com sucesso.")
except NameError as ne:
    print(f"Erro de nome: {ne}. Verifique se todas as variáveis estão definidas corretamente.")
except SyntaxError as se:
    print(f"Erro de sintaxe: {se}. Verifique a sintaxe do seu código.")
except FileNotFoundError as fnfe:
    print(f"Erro de arquivo não encontrado: {fnfe}. Verifique os caminhos dos arquivos.")
except Exception as e:
    print(f"Erro inesperado: {e}. Detalhes do erro:", type(e).__name__, e)

img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(class_names)
print('Number of classes: ', class_count)

base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_shape=img_shape,
    pooling='max'
)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    tf.keras.layers.Dense(256, kernel_regularizer=l2(0.001), activity_regularizer=l1(0.006), bias_regularizer=l1(0.006), activation='relu'),
    tf.keras.layers.Dropout(rate=0.5, seed=123),
    tf.keras.layers.Dense(class_count, activation='softmax')
])

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

print(base_model.output_shape)

model.summary()

batch_size = 32
epochs = 30
patience = 1
stop_patience = 3
threshold = 0.9
factor = 0.5
ask_epoch = 5
batches = int(tf.data.experimental.cardinality(train_ds).numpy())

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1, #0.2
    patience=3, #5
    min_lr=1e-6,
    verbose=1,
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1,
)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="./xception_model_best.keras",
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

callbacks = [early_stopping, reduce_lr, checkpoint_cb]

history = model.fit(
    x=train_ds,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=val_ds,
    validation_steps=None,
    shuffle=True
)

test_loss, test_acc, test_prec, test_rec = model.evaluate(test_ds)
print(f"\n🔎 Teste — Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

model.save('./xception_model_mri_4classes_v2.keras')

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

Y_pred = model.predict(test_ds)
y_pred = np.argmax(Y_pred, axis=1)
y_true = []
for _, labels in test_ds.unbatch():
    y_true.append(np.argmax(labels.numpy()))
y_true = np.array(y_true)
print('Confusion Matrix')
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, classes=['glioma', 'miningioma', 'pituitary', 'healthy'])
plt.show()

def plot_confusion_matrix_v2(cm, classes, normalize=True, title='Matriz de Confusão', cmap='Blues'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=classes, yticklabels=classes, square=True, cbar=True,
                annot_kws={"size": 12})
    plt.title(title)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix_v2(cm, classes=CLASSES, normalize=True)

# Binarizar y_true
y_true_bin = label_binarize(y_true, classes=list(range(len(CLASSES))))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(CLASSES)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotar curvas ROC
plt.figure(figsize=(10, 8))
for i in range(len(CLASSES)):
    plt.plot(fpr[i], tpr[i], label=f'{CLASSES[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC Multi-classe')
plt.legend(loc="lower right")
plt.grid()
plt.show()

print(classification_report(y_true, y_pred, target_names=CLASSES))