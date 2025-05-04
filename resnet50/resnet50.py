#conda activate resnet50
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef


epochs_num = 200
num_folds = 5

#train_directory="/local/data1/yanwa579/Data/resnet50_data/train"
#valid_directory="/local/data1/yanwa579/Data/resnet50_data/valid"


path_out = os.getenv("PATH_OUT")
train_directory = os.getenv("TRAIN_DIRECTORY")
valid_directory = os.getenv("VALID_DIRECTORY")
print(f"path_out is: {path_out}")
print(f"train_directory is: {train_directory}")
print(f"valid_directory is: {valid_directory}")


# If needed, change the file/figure names here
file_out_name = 'variables_output.txt'
figure_accuracy_name = 'Accuracy.pdf'
figure_loss_name = 'Loss.pdf'

# No need to change here
# Check if the directory exists
if not os.path.exists(path_out):
    # Create the directory if it doesn't exist
    os.makedirs(path_out)
    print(f"Directory '{path_out}' created.")
else:
    print(f"Directory '{path_out}' already exists.")

figure_accuracy = path_out + '/' + figure_accuracy_name
figure_loss = path_out + '/' + figure_loss_name
file_out = path_out + '/' + file_out_name


# ResNet50 model without the top layer (the dense layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the layers of the ResNet50 model to avoid training them initially
#base_model.trainable = False
for layer in base_model.layers[:15]:
    layer.trainable = False

# Data Augmentation (applies to training images only)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(factor=0.3, fill_mode='constant', fill_value=0.0),
    layers.RandomZoom(height_factor=0.1, width_factor =0.1, fill_mode='constant', fill_value=0.0),
    layers.RandomContrast(factor=0.1),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value = 0.0)
])

# Create lists of image paths and their corresponding labels
def images_and_labels(directory):
    images = []
    labels = []

    # Extract image paths and labels for training data
    for subdir in os.listdir(directory):
    #....resnet50/train/AFF,NFF
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                image = cv2.imread(file_path)
                #image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                #print(np.shape(image))
                images.append(image)
                labels.append(0 if subdir == 'NFF' else 1)
    return images,labels
train_images,train_labels = images_and_labels(train_directory)
valid_images,valid_labels = images_and_labels(valid_directory)

# Normalize
train_images = np.asarray(train_images)
valid_images = np.asarray(valid_images)
train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
train_images = train_images / 255.0
valid_images = valid_images / 255.0

#train_images = (train_images / 127.5)-1
#valid_images = (valid_images / 127.5)-1

# Mean and variance per image
#train_mean = np.mean(train_images, axis=(0, 1, 2), keepdims=True)
#train_std = np.std(train_images, axis=(0, 1, 2), keepdims=True)
#valid_mean = np.mean(valid_images, axis=(0, 1, 2), keepdims=True)
#valid_std = np.std(valid_images, axis=(0, 1, 2), keepdims=True)
#standardization
#train_images = (train_images - train_mean) / train_std
#valid_images = (valid_images - valid_mean) / valid_std


#print(type(train_images), train_images.shape, type(train_images[1,1,1,1]))
#print(type(np.array(train_labels)), np.array(train_labels).shape, type(np.array(train_labels)[1]))
#print('0:', np.max(train_images[:,:,:,0]), np.min(train_images[:,:,:,0]))
#print('1:', np.max(train_images[:,:,:,1]), np.min(train_images[:,:,:,1]))
#print('2:', np.max(train_images[:,:,:,2]), np.min(train_images[:,:,:,2]))
#print(np.array_equal(train_images[:,:,:,0], train_images[:,:,:,1]))  # True
#print(np.array_equal(train_images[:,:,:,0], train_images[:,:,:,2]))  # True
#print('u0', np.unique(train_images[:,:,:,0]))  # True
#print('u1', np.unique(train_images[:,:,:,1]))  # True
#print('u2', np.unique(train_images[:,:,:,2]))  # True


# Compute class weights using 'balanced' method
classes = np.unique(train_labels)
class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
class_weight_dict = dict(zip(classes, class_weights))


# Stop the training when the model does not improve after 25 epochs
#early_stop = EarlyStopping(monitor='val_loss', patience=25)

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

train_acc = []
val_acc = []
train_loss = []
val_loss = []

for train_idx, val_idx in kf.split(train_images):
    train_image, valid_image = train_images[train_idx], train_images[val_idx]
    train_label, valid_label = train_labels[train_idx], train_labels[val_idx]
        # Build the model
    model = models.Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')#binary classification
        ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc', curve="ROC")]#,tf.keras.metrics.F1Score(name='f1_score')]
                  )
    history = model.fit(
        x=np.array(train_image),
        y=np.array(train_label),
        batch_size=32,
        epochs=epochs_num,
        validation_data=((np.array(valid_image),np.array(valid_label))),
        class_weight=class_weight_dict,
        #callbacks=[early_stop],
        verbose=2,
        shuffle=True)

    # Store Accuracy
    train_acc.append(history.history['accuracy'])
    val_acc.append(history.history['val_accuracy'])
    train_loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
# Convert lists to numpy for easier manipulation
train_acc = np.array(train_acc)
val_acc = np.array(val_acc)
train_loss = np.array(train_loss)
val_loss = np.array(val_loss)
# Compute Mean Accuracy Over Folds
mean_train_acc = np.mean(train_acc, axis=0)
mean_val_acc = np.mean(val_acc, axis=0)
mean_train_loss = np.mean(train_loss, axis=0)
mean_val_loss = np.mean(val_loss, axis=0)

# Plot Training and Validation Accuracy
plt.figure(figsize=(8, 5))
plt.plot(mean_train_acc, label='Train Accuracy')
plt.plot(mean_val_acc, label='Validation Accuracy')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(figure_accuracy, format="pdf")

# Plot Training and Validation Auc
plt.figure(figsize=(8, 5))
plt.plot(mean_train_loss, label='Train Loss')
plt.plot(mean_val_loss, label='Validation Loss')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cross-Validation Loss')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(figure_loss, format="pdf")


result = model.evaluate(valid_images, valid_labels)
print(result)
# Print accuracy
print(f"Accuracy: {result[1]:.4f}")

# Get predicted class labels
labels_pred = model.predict(valid_images)
#print(labels_pred)
labels_pred_new = [1 if prob > 0.5 else 0 for prob in labels_pred]
#print(labels_pred)
#print(valid_labels)
# Compute F1 score,higher is better
f1 = f1_score(valid_labels, labels_pred_new,average="macro")
print(f"f1: {f1:.4f}")
# Compute Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(valid_labels, labels_pred_new)
# Compute AUC
auc = roc_auc_score(valid_labels, labels_pred)
print(f"mcc: {mcc:.4f}")
print(f"auc: {auc:.4f}")
with open(file_out, "w") as file:
    # Write variable values to file
   # file.write(f"result: {result}\n")
    file.write(f"Accuracy: {result[1]}\n")
    file.write(f"f1: {f1}\n")
    file.write(f"mcc: {mcc}\n")
    file.write(f"auc: {auc}\n")
    file.write(f"labels_pred: {[labels_pred]}\n")
    file.write(f"labels_pred_new: {labels_pred_new}\n")
    file.write(f"valid_labels: {[valid_labels]}\n")



