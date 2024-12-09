import os

# Suppress unnecessary warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import math
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall, AUC

# GPU Memory Management
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# Ensure TensorFloat-32 precision is disabled (if required)
tf.config.experimental.enable_tensor_float_32_execution(True)

# Dataset parameters
train_examples = 20225
test_examples = 2551
validation_examples = 2555
img_height = img_width = 224
batch_size = 4

# Load the ResNet50 model pre-trained on ImageNet, excluding the top layers
base_model = ResNet50(
    weights="imagenet",
    include_top=False,  # Exclude the final classification layer
    input_shape=(img_height, img_width, 3)  # Define input size
)

# Freeze the base model
base_model.trainable = False

# Add custom layers on top of ResNet50
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Pool the features from the base model
    layers.Dense(256, activation="relu"),  # Fully connected layer
    layers.Dropout(0.5),  # Add dropout for regularization
    layers.Dense(1, activation="sigmoid"),  # Final binary classification layer
])

# Data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Use ResNet50's preprocessing
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Data generators
train_gen = train_datagen.flow_from_directory(
    "data/train/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)


validation_gen = validation_datagen.flow_from_directory(
    "data/validation/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

test_gen = test_datagen.flow_from_directory(
    "data/test/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

# Compile the model with updated optimizer parameter
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),  # Updated to use `learning_rate`
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        "accuracy",  # Accuracy metric
        Precision(name="precision"),  # Precision metric
        Recall(name="recall"),  # Recall metric
        AUC(name="auc")  # AUC metric
    ]
)

# Train the model
model.fit(
    train_gen,
    epochs=1,
    steps_per_epoch=train_examples // batch_size,
    validation_data=validation_gen,
    validation_steps=validation_examples // batch_size,
)

# model.fit(
#     train_gen,
#     steps_per_epoch=len(train_gen) // batch_size,
#     epochs=1,
#     validation_data=validation_gen,
#     validation_steps=validation_examples // batch_size,
# )
# ROC Curve Plot
# Predict on the entire test set
predictions = model.predict(test_gen, steps=math.ceil(test_examples / batch_size))

# Collect true labels
test_labels = []
for _, y_batch in test_gen:
    test_labels.extend(y_batch.flatten())
    if len(test_labels) >= test_examples:
        break
test_labels = np.array(test_labels[:test_examples])  # Truncate if necessary

# Compute ROC curve
fp, tp, _ = roc_curve(test_labels, predictions[:test_examples])

# Plot ROC curve
plt.plot(100 * fp, 100 * tp)
plt.xlabel("False positives [%]")
plt.ylabel("True positives [%]")
plt.title("ROC Curve")
plt.show()

# Evaluate the model
model.save("resnet_model")
model.evaluate(validation_gen, verbose=2)
model.evaluate(test_gen, verbose=2)
