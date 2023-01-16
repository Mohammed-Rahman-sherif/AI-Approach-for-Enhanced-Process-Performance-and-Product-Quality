#attention score
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

path = '/mnt/DataStore/Sherif/mode_extended_time_range_tw_21600_rp_600/mode_extended_time_range_tw_21600_rp_600'

def load_preprocessed_parquet(PATH: str, FOLD: str):
    full_array = [np.load(PATH + "/" + FOLD + "/x_train.npy"),
                  np.load(PATH + "/" + FOLD + "/x_val.npy"),
                  np.load(PATH + "/" + FOLD + "/y_train.npy"),
                  np.load(PATH + "/" + FOLD + "/y_val.npy"),
                  np.load(PATH + "/" + FOLD + "/full_train_x.npy"),
                  np.load(PATH + "/" + FOLD + "/full_train_y.npy")]
    return full_array

data = load_preprocessed_parquet(path, 'fold-2')

x_train = data[0]
y_train = data[2]
x_test = data[1]
y_test = data[3]


n_classes = len(np.unique(y_train))
print(n_classes)
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

n_timesteps = x_train.shape[1]
print('x_train:', x_train.shape)
n_features = x_train.shape[2]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout, return_attention):
    if return_attention:
        x,y = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout,
        )(inputs, inputs, return_attention_scores=True) 
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res, y

    else:
        print(type(head_size), type(num_heads), type(ff_dim))
        print('inputs', inputs)
        x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

    # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        print('block', _)
        if _ != 1 and _ != 6:
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, return_attention=False)
        else:
            x, attn = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, return_attention=True)
    z = inputs
    for _ in range(num_transformer_blocks):
        if _ != 1 and _ != 6:
            z = transformer_encoder(z, head_size, num_heads, ff_dim, dropout, return_attention=False)
        else:
            z, attn = transformer_encoder(x, head_size, num_heads, ff_dim, dropout, return_attention=True)

    z = inputs

    y = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    z = layers.GlobalAveragePooling1D(data_format="channels_first")(z)
    x = layers.Concatenate(axis=1)([z, y])
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

input_shape = (n_timesteps, n_features)
model = build_model(
    input_shape,
    head_size=128,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=6,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [
 keras.callbacks.CSVLogger('/mnt/DataStore/Sherif/resample_10_6h/Log_Files/new5/log0.csv', separator = ',', append = True),
 keras.callbacks.ModelCheckpoint('/mnt/DataStore/Sherif/resample_10_6h/transformer_models/new5/model0.{epoch:02d}-{val_sparse_categorical_accuracy:.2f}.h5', monitor = 'val_sparse_categorical_accuracy', varbose = 0, save_best_only = True, mode = 'max')
]

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=300,
    batch_size=64,
    callbacks = callbacks)
print(history.history)
model.evaluate(x_test, y_test, verbose=1)
#model.save('/mnt/DataStore/Sherif/transformer_models/model0.h5')
model.save_weights('/mnt/DataStore/Sherif/resample_10_6h/transformer_weights/new5/weight0.h5')

plt.plot(history.history['sparse_categorical_accuracy'])
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Acc Value')
plt.xlabel('No of Epochs')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()
plt.savefig('/mnt/DataStore/Sherif/resample_10_6h/transformer_graphs/new5/figure0_accuracy.png')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss Value')
plt.xlabel('No of Epochs')
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()
plt.savefig('/mnt/DataStore/Sherif/resample_10_6h/transformer_graphs/new5/figure0_loss.png')


predictions = model.predict(x_test)

y_pred = (predictions)
#print(y_pred)
np.save('f0y_pred', y_pred)
np.save('f0x_train', x_train)
np.save('f0y_train', y_train)
np.save('f0x_test', x_test)
np.save('f0y_test', y_test)
