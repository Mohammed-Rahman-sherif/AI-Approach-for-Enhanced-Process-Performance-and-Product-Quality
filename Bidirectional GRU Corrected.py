import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.metrics import confusion_matrix
#path to data
path = 'C:\\Users\SashaMK3\Desktop\Work Project\Data\mode_extended_time_range_tw_21600_rp_600 (1)\mode_extended_time_range_tw_21600_rp_600\mode_extended_time_range_tw_21600_rp_600'

# hyperparams
learning_rate = 0.001
batch_size = 1
epochs = 100
layers = 64

# Loading the data from the path
def load_preprocessed_parquet(PATH: str, FOLD: str):
    full_array = [np.load(PATH + "/" + FOLD + "/x_train.npy"),
                  np.load(PATH + "/" + FOLD + "/y_train.npy"),

                  np.load(PATH + "/" + FOLD + "/x_val.npy"),
                  np.load(PATH + "/" + FOLD + "/y_val.npy")]
    return full_array

# loads all of the folds and gets the training data from fold 0 and the val data from all folds
data = load_preprocessed_parquet(path, 'fold-0')
testdata1 = load_preprocessed_parquet(path, 'fold-1')
testdata2 = load_preprocessed_parquet(path, 'fold-2')
testdata3 = load_preprocessed_parquet(path, 'fold-3')
testdata4 = load_preprocessed_parquet(path, 'fold-4')
# splitting the data into different arrays

#fold 0
x_train = data[0]
y_train = data[1]
x_val = data[2]
y_val = data[3]
# fold 1
x_train1 = testdata1[0]
y_train1 = testdata1[1]
x_val1 = testdata1[2]
y_val1 = testdata1[3]
# fold 2
x_train2 = testdata2[0]
y_train2 = testdata2[1]
x_val2 = testdata2[2]
y_val2 = testdata2[3]
# fold 3
x_train3 = testdata3[0]
y_train3 = testdata3[1]
x_val3 = testdata3[2]
y_val3 = testdata3[3]
# fold 4
x_train4 = testdata4[0]
y_train4 = testdata4[1]
x_val4 = testdata4[2]
y_val4 = testdata4[3]

for x in range(5):
    if x == 0:
        x_train = x_train
        y_train = y_train
        x_val = x_val
        y_val = y_val
    elif x == 1:
        x_train = x_train1
        y_train = y_train1
        x_val = x_val1
        y_val = y_val1
    elif x == 2:
        x_train = x_train2
        y_train = y_train2
        x_val = x_val2
        y_val = y_val2
    elif x == 3:
        x_train = x_train3
        y_train = y_train3
        x_val = x_val3
        y_val = y_val3
    elif x == 4:
        x_train = x_train4
        y_train = y_train4
        x_val = x_val4
        y_val = y_val4

    print('Fold '+ str(x))
    # getting the input data
    inputs = keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    # creating the model
    model = tf.keras.Sequential()
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(layers, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(layers, return_sequences=True)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(2))
    model.add(keras.layers.Activation('softmax'))

    # compiling the model monitoring loss and accuracy per epoch
    model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=["sparse_categorical_accuracy"])

    # shows the model summary
    model.summary()
    # stops training when the model stops improving
    earlyStop = EarlyStopping(monitor='sparse_categorical_accuracy', patience=5, restore_best_weights=True)


    # fits the model and evaluates it performance based on the validation data giving
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[earlyStop],
        # accuracy and loss shown at the end of every epoch based on this data
       # validation_data=(x_val, y_val),
    )
    print("validation data:")
    model.evaluate(x_val, y_val, batch_size=batch_size)

    predictions = model.predict(x_val)
    y_pred = np.around(predictions)
    # reshapes the array to only show one set of predictions
    y_pred = y_pred[:, 1]
    if x == 0:
        matrix = confusion_matrix(y_val, y_pred)
        print('saved 0 ')
    elif x == 1:
        matrix1 = confusion_matrix(y_val, y_pred)
        print('saved1')
    elif x == 2:
        matrix2 = confusion_matrix(y_val, y_pred)
        print('saved2')
    elif x == 3:
        matrix3 = confusion_matrix(y_val, y_pred)
        print('saved3')
    elif x == 4:
        matrix4 = confusion_matrix(y_val, y_pred)
        print('saved4')

for y in range(6):
    if y == 0:
        matrix =matrix
    elif y == 1:
        matrix =matrix1
    elif y == 2:
        matrix =matrix2
    elif y == 3:
        matrix =matrix3
    elif y == 4:
        matrix =matrix4
    elif y == 5:
        TL = matrix[0, 0] + matrix1[0, 0] + matrix2[0, 0] + matrix3[0, 0] + matrix4[0, 0]

        TR = matrix[0, 1] + matrix1[0, 1] + matrix2[0, 1] + matrix3[0, 1] + matrix4[0, 1]

        BL = matrix[1, 0] + matrix1[1, 0] + matrix2[1, 0] + matrix3[1, 0] + matrix4[1, 0]

        BR = matrix[1, 1] + matrix1[1, 1] + matrix2[1, 1] + matrix3[1, 1] + matrix4[1, 1]

        matrix = np.array([[TL, TR], [BL, BR]])

    ax = sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g')
    ax.set_title('Confusion Matrix LSTM Layers = ' + str(layers) + '\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    # Display the visualization of the Confusion Matrix.
    plt.show()



