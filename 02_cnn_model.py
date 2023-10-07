import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils.vis_utils import plot_model

#Tensorflow verzió kiírása
print(tf. __version__)
print(tf.config.list_physical_devices('GPU'))

#Az adathalmazból készített JSON fájl elérési útja
DATASET = "data_10segments.json"

#Adathalmazok előkészítése, train, test és validation split készítése
def prepare_datasets(test_size, validation_size):
    print("Adathalmaz betöltése.")
    # Adatok betöltése
    X, y, genre_dict = load_dataset(DATASET)
    print("Training, teszt és validációs halmazok létrehozása.")
    # train és test halmaz készítése
    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=test_size)

    # train és validation halmaz készítése
    X_train_data, X_validation_data, y_train_data, y_validation_data = train_test_split(X_train_data, y_train_data, test_size=validation_size)

    # 3 dimenziós tömb -> alakja: (130, 13 (mfccs), 1 (channels))
    X_train_data = X_train_data[..., np.newaxis] # 4 dimenziós tömbbé alakítás -> (num_samples, 130, 13, 1)
    X_validation_data = X_validation_data [..., np.newaxis]
    X_test_data = X_test_data[..., np.newaxis]
    print("Előkészítés sikeres.")
    return X_train_data, X_validation_data, X_test_data, y_train_data, y_validation_data, y_test_data, genre_dict

#Az adathalmaz betöltése
def load_dataset(dataset_path):
    print("Adathalmaz megnyitása.")
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    X = np.array(data["mfcc_vectors"]) # bemenetek
    y = np.array(data["index"]) # kimenetek

    print("Bemenetek és kimenetek átalakítása numpy tömbbé sikeres!")

    genre_dict = np.array(data["genres"])

    print("Elérhető műfajok: "+ str(genre_dict))

    return X, y, genre_dict

#Neurális hálózat model felépítése
def create_cnn_model(input_shape):
    num_of_outputs = len(genre_dict)
    print(str(num_of_outputs))
    # Model elkészítése
    model = tf.keras.Sequential()

    print("Konvolúciós rétegek létrehozása")

    # 1. Konvolúciós réteg
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)) # (Kernelek száma, (Kernelek grid/rács nagysága), Aktivációs fgv típusa, Bemenet alakja)
    model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 2. Konvolúciós réteg
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 3. Konvolúciós réteg
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # kimenet lapítása (flatten) és tovább adása a Dense rétegnek
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    print("Kimeneti réteg létrehozása")

    # Kimeneti réteg
    model.add(tf.keras.layers.Dense(num_of_outputs, activation='softmax')) # neuronok száma = műfajok száma = num_of_outputs

    print("A Konvolúciós model sikeresen létrejött.")

    return model

def optimizer():
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def create_diagram():
    if not os.path.isfile('Model_Diagram.png'):
        plot_model(model, to_file='Model_Diagram.png', show_shapes=True, show_layer_names=True, expand_nested=True)
        print("A neurális háló felépítélsének diagramja sikeresen elkészült!")

#Predikció
def make_prediction(model, X, y):
    print("Predikció a teszt halmazból.")
    X = X[np.newaxis, ...]
    # Predikció = [ [0.1, 0.2, ...] ]
    prediction = model.predict(X) # X -> (130, 13, 1) model.predict 4 dimenziós tömböt vár ezért X -> (1, 130, 13, 1)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print("Elvárt műfaj: {}, Predikció: {}".format(genre_dict[int(y)], genre_dict[int(predicted_index)]))

if __name__ == "__main__":
    # Training, validáció és teszt halmazok készítése
    X_train_data, X_validation_data, X_test_data, y_train_data, y_validation_data, y_test_data, genre_dict = prepare_datasets(0.25, 0.2)
    # CNN háló bemenete
    input_shape = (X_train_data.shape[1], X_train_data.shape[2], X_train_data.shape[3])
    print("Bemenet alakja (Input shape): "+ str(input_shape))
    model = create_cnn_model(input_shape)

    # Hálózat optimalizálása
    optimizer()

    #CNN összefoglalás
    model.summary()
    print(genre_dict)

    #Model diagramjanak elkeszitese
    create_diagram()

    # CNN tanítésa
    print("A model betanítása:")
    savemodel = model.fit(X_train_data, y_train_data, validation_data=(X_validation_data, y_validation_data), batch_size=32, epochs=600)
    # Kiértékelés a teszt adatokon
    test_error, test_accuracy = model.evaluate(X_test_data, y_test_data, verbose=1)
    print("Pontosság a teszt halmazon: {}".format(test_accuracy))
    model.save('model.h5', savemodel)
    print("Model sikeresen lementve!")

    # Predikció egy mintán
    Prediction_X = X_test_data[50]
    Prediction_y = y_test_data[50]

    make_prediction(model, Prediction_X, Prediction_y)
