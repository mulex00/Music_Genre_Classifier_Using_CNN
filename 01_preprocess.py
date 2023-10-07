import os
import librosa
import math
import json
from zipfile import ZipFile

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# A letöltött dataset elérési útja
DATASET = "Data/genres_original"
# A program által készített JSON fájl neve
JSON = "data_10segments.json"

def download_and_unzip_dataset():
    print("Kérem várjon amíg az adathalmaz letöltődik!")

    #Dataset letoltese Kaggle API kapcsolattal
    try:
        api.dataset_download_files("andradaolteanu/gtzan-dataset-music-genre-classification")
        print("Sikeres letöltés")
    except Exception as e:
        print("A letöltés során hiba lépett fel. Kérem próbálja újra! Hibaüzenet:")
        print(e)

    print("Kérem várjon amíg a letöltött adathalmazt kicsomagoljuk!")

    #Letoltott Dataset kicsomagolasa
    try:
        zf = ZipFile('gtzan-dataset-music-genre-classification.zip')
        zf.extractall()
        zf.close()
        # Korrupt fájl törlése jazz.00054.wav
        os.remove("Data/genres_original/jazz/jazz.00054.wav")
        print("A kicsomagolás sikeres!")
    except Exception as e:
        print("A kicsomagolás során hiba lépett fel. Kérem próbálja újra! Hibaüzenet:")
        print(e)

SAMPLE_RATE = 22050 # Audió fájlok mintavételezési frekvenciája
AUDIO_DURATION = 30 # másodpercben mérve
SAMPLES = SAMPLE_RATE * AUDIO_DURATION # Minták száma egy audió fájlon belül

def create_mfcc(dataset_location, json_location, num_of_segments):

    #Fast Fourier transzformációk száma
    num_of_fft = 2048
    #MFCC együtthatók száma
    num_of_mfcc = 13

    # JSON-ba tárolt adatok címkéi
    data = {
        "genres": [], # műfajok például classical, blues stb...
        "mfcc_vectors": [], # training bemenetek
        "index": [] # műfajok indexe
    }

    number_of_samples_per_segment = int(SAMPLES / num_of_segments)
    hop_length = 512
    expected_number_of_mfcc_vectors = math.ceil(number_of_samples_per_segment / hop_length) # pl: 1.2 -> 2

    # Végigmegy a műfajok mappáin
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_location)):

        # A jelenlegi elérési út, már nem a megadott útvonal
        if dirpath is not dataset_location:

            # műfaj elnevezések elmentése mappa név alapján
            dirpath_components = dirpath.split("\\")
            current_genre = dirpath_components[-1]
            data["genres"].append(current_genre)
            print("\nFeldolgozás alatt: {}".format(current_genre))

            # Fájlok feldolgozása
            for f in filenames:

                # Audió fájl betöltése
                file = os.path.join(dirpath, f)
                signal, sr = librosa.load(file, sr=SAMPLE_RATE)

                # Audió fájl szegmensekre vágása és mfcc-vé alakítása
                for s in range(num_of_segments):
                    start_sample = number_of_samples_per_segment * s
                    finish_sample = start_sample + number_of_samples_per_segment

                    # MFCC készítése librosaval
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=num_of_fft,
                                                n_mfcc=num_of_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # Ha az mfcc hossza megegyezik a várt mfcc hosszával, akkor elmenti
                    if len(mfcc) == expected_number_of_mfcc_vectors:
                        data["mfcc_vectors"].append(mfcc.tolist())
                        data["index"].append(i-1)
                        print("{}, szegmens:{}".format(file, s+1))

    # Kiírás JSON fájlba
    with open(json_location, "w") as fp:
        json.dump(data, fp, indent=4)
        print("Az előfeldolgozás sikeres! A JSON fájl sikeresen létrejött!")

if __name__ == "__main__":
    if os.path.exists(DATASET):
        print("Az adathalmaz elérhető a könyvtárban!")
        create_mfcc(DATASET, JSON, num_of_segments=10)
    else:
        print("Az adathalmaz nem érhető el a könyvtárban!")
        download_and_unzip_dataset()
        create_mfcc(DATASET, JSON, num_of_segments=10)