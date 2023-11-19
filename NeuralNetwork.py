# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#import tensorflow as tf
#from kerastuner.tuners import RandomSearch
#from kerastuner.engine.hyperparameters import HyperParameter
import numpy as np
from keras.datasets import mnist
from keras import datasets, layers, models
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from RMDL import RMDL_Image

(x_train,y_train),(x_test,y_test) = mnist.load_data()

# Split the data into training and temporary sets (combining validation and test)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

# Further split the validation set into validation and test sets
x_val, x_test, y_val, y_test = train_test_split(
    x_val, y_val, test_size=0.5, random_state=42)

# Optionally, normalize pixel values to the range [0, 1]
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0
x_train_reshaped = x_train.reshape((x_train.shape[0], 28, 28, 1)) / 255.0
x_val_reshaped = x_val.reshape((x_val.shape[0], 28, 28, 1)) / 255.0
x_test_reshaped = x_test.reshape((x_test.shape[0], 28, 28, 1)) / 255.0

#DNN
modelDNN = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'), # Ookreślenie liczby neuronów i funkcji aktywacji
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

modelDNN.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#CNN
modelCNN = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
#RMDL
modelRMDL = RMDL_Image.Image_Classification(x_train, y_train, x_test, y_test,(28,28,1),
                             batch_size=128,
                             sparse_categorical=True,
                             epochs=3)


modelCNN.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])



#Training

trainHistory = modelDNN.fit(x_train, y_train, epochs=3,validation_data=(x_val, y_val))
trainHistoryCnn = modelCNN.fit(x_train, y_train, epochs=3, validation_data=(x_val, y_val))
#rmdl_model.fit()
#Model evaluation
y_pred = modelDNN.predict(x_test)
y_pred_classes = np.argmax(y_pred,axis=1)
y_pred_cnn = modelCNN.predict(x_test)
y_pred_classes_cnn = np.argmax(y_pred_cnn, axis=1)
#y_pred_rmdl = rmdl_model.predict(np.array(x_test))


#Metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
accuracy_cnn = accuracy_score(y_test, y_pred_classes_cnn)
precision_cnn = precision_score(y_test, y_pred_classes_cnn, average='weighted')
recall_cnn = recall_score(y_test, y_pred_classes_cnn, average='weighted')
#accuracy_rmdl = accuracy_score(y_test, y_pred_rmdl)
#precision_rmdl = precision_score(y_test, y_pred_rmdl, average='weighted')
#recall_rmdl = recall_score(y_test, y_pred_rmdl, average='weighted')


print(f'Dokładność klasyfikacji: {accuracy}')
print(f'Precyzja: {precision}')
print(f'Czułość: {recall}')
print(f'Dokładność klasyfikacji (CNN): {accuracy_cnn}')
print(f'Precyzja (CNN): {precision_cnn}')
print(f'Czułość (CNN): {recall_cnn}')
#print(f'Dokładność klasyfikacji (RMDL): {accuracy_rmdl}')
#print(f'Precyzja (RMDL): {precision_rmdl}')
#print(f'Czułość (RMDL): {recall_rmdl}')
# Compute ROC curve and AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
y_pred_bin = label_binarize(y_pred_classes, classes=np.unique(y_test))

fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC')
plt.legend()
plt.show()





















# Not working optimalization
# def build_model(hp):
#     model = models.Sequential()
#     model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
#                     activation='relu', input_shape=(28, 28)))
#     model.add(layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
#     model.add(layers.Dense(10, activation='softmax'))
#
#     model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     return model
#
#
# tuner = RandomSearch(build_model,
#                      objective='val_accuracy',
#                      max_trials=5,
#                      executions_per_trial=3,
#                      directory='my_dir',
#                      project_name='NeuralNetwork')
#
# tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
#
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#
# # Display the best hyperparameters
# print(f"Best Hyperparameters: {best_hps.values}")