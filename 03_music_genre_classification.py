import math
import json
import librosa.feature, librosa.display
from pydub.utils import mediainfo
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pygame
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from mutagen.id3 import ID3
from mutagen.flac import FLAC
from mutagen.wave import WAVE
from tinytag import TinyTag
import sys
import os

#Tensorflow verzió ellenörzés
print(tf. __version__)
print(tf.config.list_physical_devices('GPU'))

#Dataset elérése és műfaj kategóriák
DATASET_PATH = "data_10segments.json"
with open(DATASET_PATH, "r") as fp:
    data = json.load(fp)
    genre_dict = np.array(data["genres"])

predicted_genre = " "
predicted_genre_accuracy = " "

def predict_new_song(model, X):
    X = X[np.newaxis, ..., np.newaxis]

    # prediction = [ [0.1, 0.2, ...] ]
    prediction = model.predict(X) # X -> (130, 13, 1) but model.predict expects 4D array so X -> (1, 130, 13, 1)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    return predicted_index


def process_input(audio_file,):

    dur = float(mediainfo(audio_file)["duration"])

    SAMPLE_RATE = 22050
    NUM_MFCC = 13
    N_FTT = 2048
    TRACK_DURATION = float(dur)
    HOP_LENGTH = 512
    NUM_SEGMENTS = math.floor(dur/3)
    print("Szegmensek száma: "+ str(NUM_SEGMENTS))
    SAMPLES_PER_TRACK = SAMPLE_RATE * NUM_SEGMENTS*3 # SAMPLE RATE * TRACK DURATION

    samples_per_segment = math.ceil(int(SAMPLES_PER_TRACK / NUM_SEGMENTS))
    expected_number_of_mfcc_vectors = math.ceil(samples_per_segment / HOP_LENGTH) # ((22050*30)/10)/512 = 129,199219 (felfele kerekitve 130 ami az expected shape)

    signal, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=math.floor(float(dur)))
    mfcc =""
    mfcc_array = []
    for d in range(NUM_SEGMENTS):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=SAMPLE_RATE, n_mfcc=NUM_MFCC, n_fft=N_FTT,hop_length=HOP_LENGTH )
        mfcc = mfcc.T

        mfcc_array.append(mfcc)

        print("MFCC")
        print(mfcc_array[d].shape)

    return mfcc_array, NUM_SEGMENTS, mfcc

def pred(songname):
    model = tf.keras.models.load_model('model.h5')

    new_input_mfcc = process_input(songname)
    prediction_array = []
    for i in range(new_input_mfcc[1]):

        X = new_input_mfcc[0][i]
        prediction = predict_new_song(model, X)
        print(str(i+1)+". szegmens: "+ str(genre_dict[int(prediction)]))
        prediction_array.append(prediction)

    unique, counts = np.unique(prediction_array, return_counts=True)
    print(counts)
    index = np.argmax(counts)
    global predicted_genre
    predicted_genre = genre_dict[int(unique[index])]
    print(predicted_genre)
    global predicted_genre_accuracy
    predicted_genre_accuracy = str(counts[index]) + " / " + str(len(prediction_array)) + " szegmensből"

class MyWindow(QMainWindow):

    stime = None
    elapsed = 0
    # PyQt5 Design
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi("music_genre_classifier_UI.ui", self)
        self.show()
        self.initUI()

    def initUI(self):
        #Fájl megnyitása
        self.OpenButton.clicked.connect(self.open_file)

        #Plotok kirajzolásához gombok
        #Hullámforma megjelenítése
        self.ShowWaveformButton.clicked.connect(self.show_waveform)
        #Spektrum megjelenítése
        self.ShowSpectrumButton.clicked.connect(self.show_spectrum)
        #Spektogram megjeleítése
        self.ShowSpectogramButton.clicked.connect(self.show_spectogram)
        #MFCC megjeleítése
        self.ShowMFCCButton.clicked.connect(self.show_MFCC)

        #Prediction
        self.PredictButton.clicked.connect(self.pred_file)

        #Zenelejátszó gombjai
        self.PlayMusicButton.setStyleSheet("QPushButton{border-image: url(images/playbutton.png); border: 0px;}QPushButton:hover{border-image: url(images/playbutton_pressed.png); border: 0px;}")
        #self.PlayMusicButton.setText("Play")
        #self.PlayMusicButton.move(256, 528)
        self.PlayMusicButton.clicked.connect(self.play_music)
        self.StopMusicButton.clicked.connect(self.stop_music)
        self.ForwardMusicButton.clicked.connect(self.forward_music)
        self.BackwardMusicButton.clicked.connect(self.backward_music)

        #Media player elkészítése
        pygame.mixer.init()

        #Coverart a zenéhez
        pixmap = QPixmap('images/albumcover.png')
        self.label_coverart.setPixmap(pixmap.scaled(320, 320))
        self.label_coverart.adjustSize()

        #Predicted Genre
        self.label_predictedgenre.setText(predicted_genre)
        self.label_predictedgenre.adjustSize()

        #Predicted Genre Accuracy
        self.label_predictedgenreaccuracy.setText(predicted_genre_accuracy)
        self.label_predictedgenreaccuracy.adjustSize()


    #Fájl megnyitása
    def open_file(self):
        self.played = False
        self.filepath = QFileDialog.getOpenFileName(self, 'Válassz ki egy audió fájlt!', '.', 'Audio Files (*.wav *.flac *.mp3)')
        # Teszt nem megfelelő fájlformátum esetén
        print(self.filepath[0])
        self.fileformat = "unknown"
        if (self.filepath[0]) and (self.filepath[0].lower().endswith(('.wav','.flac','.mp3'))):
            print("hello")
            self.opened_file = self.filepath[0]
            #Előzőleg megnyitott zenék tag-jeinek eltüntetése
            self.label_songtitle.setText("Cím: ")
            self.label_songartist.setText("Előadó: ")
            self.label_songrelease.setText("Kiadás éve: ")
            self.label_songbitrate.setText("Bitráta: ")

            #Betöltés a mediaplayerbe
            pygame.mixer.music.load(self.opened_file)
            # Albumborító kinyerése
            try:
                #Audio metadata tag-jeinek kinyerése
                audiotag_details = TinyTag.get(self.filepath[0])

                if audiotag_details.title:
                    songtitle = audiotag_details.title
                    self.label_songtitle.setText("Cím: "+ songtitle)
                else:
                    self.label_songtitle.setText("Cím: ")

                if audiotag_details.artist:
                    songartist = audiotag_details.artist
                    self.label_songartist.setText("Előadó: "+ songartist)
                else:
                    self.label_songartist.setText("Előadó: ")
                if audiotag_details.year:
                    songrelease = audiotag_details.year
                    self.label_songrelease.setText("Kiadás éve: "+ songrelease)
                else:
                    self.label_songrelease.setText("Kiadás éve: ")
                if audiotag_details.bitrate:
                    songbitrate = audiotag_details.bitrate
                    self.label_songbitrate.setText("Bitráta: "+ str(round(songbitrate))+ " kBits/s" )
                else:
                    self.label_songbitrate.setText("Bitráta: ")
            except Exception as e:
                print(e)
            #Albumborító megjelenítése különböző formátumok esetén. Ha nincs albumborító, akkor audiotag="none" marad.
            audiotag="none"
            #MP3 fájlformátum
            try:
                if ID3(self.filepath[0]):
                    audio = ID3(self.filepath[0])
                    for tag in audio:
                        if tag.startswith("APIC"):
                            if audio[tag].data:
                                pic = audio[tag].data
                                with open("temp.jpg", "wb") as f:
                                    f.write(pic)
                                pixmap = QPixmap('temp.jpg')
                                os.remove("temp.jpg")
                                audiotag = "ID3"
            except Exception as e:
                print(e)

            #FLAC fájlformátum
            try:
                if FLAC(self.filepath[0]):
                    audio = FLAC(self.filepath[0])
                    audiotag = "FLAC"
                    if audio.pictures:
                        pics = audio.pictures
                        for p in pics:
                            if p.type==3:
                                pic = p.data
                                with open("temp.jpg", "wb") as f:
                                    f.write(pic)
                                pixmap = QPixmap('temp.jpg')
                                os.remove("temp.jpg")
                    else:
                        pixmap = QPixmap('images/albumcover.png')
            except Exception as e:
                print(e)

            #WAV fájlformátum
            try:
                if WAVE(self.filepath[0]):
                    audio = WAVE(self.filepath[0])
                    audiotag = "WAVE"
                    for tag in audio:
                        if tag.startswith("APIC"):
                            if audio[tag].data:
                                pic = audio[tag].data
                                with open("temp.jpg", "wb") as f:
                                    f.write(pic)
                                pixmap = QPixmap('temp.jpg')
                                os.remove("temp.jpg")
            except Exception as e:
                print(e)
            try:
                if audiotag == "none":
                    pixmap = QPixmap('images/albumcover.png')
            except Exception as e:
                print(e)

            #fájl nevének kinyerése az elérési útvonalból
            self.filename = os.path.basename(self.opened_file)
            if(str(self.filename).lower().endswith(('.wav','.flac','.mp3'))):
                print('Kiválasztott fájl: ', self.filename)

            global predicted_genre
            predicted_genre = " "
            global predicted_genre_accuracy
            predicted_genre_accuracy = " "

            self.label_predictedgenre.setText(predicted_genre)
            self.label_predictedgenreaccuracy.setText(predicted_genre_accuracy)
            self.label_filename.setText(self.filename)
            self.label_filename.adjustSize()
            self.label_coverart.setPixmap(pixmap.scaled(320, 320))
            self.update()
        else:
            print("Hiba történt a fájl megnyitása során!")

    #LIBROSA funkciók
    def show_waveform(self):
        try:
            file = self.opened_file
            #Hanghullam hossza
            dur = mediainfo(file)["duration"]
            #Fajl megadasa librosanak
            signal, sr = librosa.load(file, sr=None, duration=math.floor(float(dur)))
            #Hullamforma letrehozasa
            librosa.display.waveshow(signal, sr=sr, alpha=0.5, color='b')
            #Plot elkeszitese
            plt.title(self.filename)
            plt.xlabel("Idő")
            plt.ylabel("Amplitúdó")
            plt.show()
        except Exception as e:
            print(e)

    def show_spectrum(self):
        try:
            file = self.opened_file
            #Spektrum
            dur = mediainfo(file)["duration"]
            signal, sr = librosa.load(file, sr=22050, duration=math.floor(float(dur))) # sr(SampleRate) * T(duration)
            #Fast Fourier Transzformacio
            fft = np.fft.fft(signal)
            #Hang intenzitas es frekvencia megadasa
            magnitude = np.abs(fft)
            frequency = np.linspace(0, sr, len(magnitude))  #Egyenletes tavolsagban levo szamok szama egy intervallumban
            left_frequency = frequency[:int(len(frequency) / 2)]
            left_magnitude = magnitude[:int(len(frequency) / 2)]

            plt.plot(left_frequency, left_magnitude)
            plt.title(self.filename)
            plt.xlabel("Frekvencia")
            plt.ylabel("Magnitude")
            plt.show()
        except Exception as e:
            print(e)

    def show_spectogram(self):
        try:
            file = self.opened_file
            #Spektogram
            dur = mediainfo(file)["duration"]
            signal, sr = librosa.load(file, sr=None, duration=math.floor(float(dur))) # sr(SampleRate) * T(duration)
            n_fft = 2048  #Mintak szama
            hop_length = 512  #Eltolas jobbra

            stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
            spectogram = np.abs(stft)

            log_spectogram = librosa.amplitude_to_db(spectogram)  #Amplitudo atalakitasa dB-re

            librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)

            plt.title(self.filename)
            plt.xlabel("Idő")
            plt.ylabel("Frekvencia")
            plt.colorbar(format="%+ 2.0f dB")
            plt.show()
        except Exception as e:
            print(e)

    def show_MFCC(self):
        try:
            file = self.opened_file
            #MFCC
            dur = mediainfo(file)["duration"]
            signal, sr = librosa.load(file, sr=None, duration=math.floor(float(dur))) # sr(SampleRate) * T(duration)
            n_fft = 2048  #Mintak szama
            hop_length = 512  #Eltolas jobbra

            MFCCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length,
                                         n_mfcc=13)

            librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length, x_axis="time")

            plt.title(self.filename)
            plt.xlabel("Idő")
            plt.ylabel("MFCC")
            plt.colorbar()
            plt.show()
        except Exception as e:
            print(e)

    #Kiválasztott fájl műfajának meghatározása
    def pred_file(self):
        try:
            filepath = self.opened_file
            pred(filepath)
            self.label_predictedgenre.setText(predicted_genre)
            self.label_predictedgenreaccuracy.setText(predicted_genre_accuracy)
            self.update()
        except Exception as e:
            print(e)
            self.open_file()

    #Zenelejátszó funkciók
    def play_music(self):
        try:
            global stime, elapsed
            now = time.time()
            if (pygame.mixer.music.get_busy() == False):
                if (self.played == True):
                    pygame.mixer.music.unpause()
                    self.PlayMusicButton.setStyleSheet("QPushButton{border-image: url(images/pausebutton.png); border: 0px;}QPushButton:hover{border-image: url(images/pausebutton_pressed.png); border: 0px;}")
                    stime = now - elapsed
                else:
                    pygame.mixer.music.play(loops=0)
                    self.PlayMusicButton.setStyleSheet("QPushButton{border-image: url(images/pausebutton.png); border: 0px;}QPushButton:hover{border-image: url(images/pausebutton_pressed.png); border: 0px;}")
                    stime = time.time()
                    pygame.mixer.music.set_pos(0)
                    self.played = True
            else:
                now = time.time()
                elapsed = now - stime
                pygame.mixer.music.pause()
                self.PlayMusicButton.setStyleSheet("QPushButton{border-image: url(images/playbutton.png); border: 0px;}QPushButton:hover{border-image: url(images/playbutton_pressed.png); border: 0px;}")
        except Exception as e:
            print(e)

    def stop_music(self):
        try:
            pygame.mixer.music.stop()
            self.played = False
            self.PlayMusicButton.setStyleSheet("QPushButton{border-image: url(images/playbutton.png); border: 0px;}QPushButton:hover{border-image: url(images/playbutton_pressed.png); border: 0px;}")
        except Exception as e:
            print(e)

    def forward_music(self):
        try:
            global stime, elapsed
            if stime and pygame.mixer.music.get_busy() == True:
                elapsed = time.time() - stime
                delta = min(elapsed, 5)
                pygame.mixer.music.play(start=elapsed + delta)
                stime -= delta
        except Exception as e:
            print(e)

    def backward_music(self):
        try:
            global stime, elapsed
            if stime and pygame.mixer.music.get_busy() == True:
                elapsed = time.time() - stime
                delta = min(elapsed, 5)
                pygame.mixer.music.play(start=elapsed - delta)
                stime += delta
        except Exception as e:
            print(e)

    #Labelek frissítése
    def update(self):
        self.label_title.adjustSize()
        self.label_songtitle.adjustSize()
        self.label_songartist.adjustSize()
        self.label_songrelease.adjustSize()
        self.label_songbitrate.adjustSize()
        self.label_coverart.adjustSize()
        self.label_predictedgenre.adjustSize()
        self.label_predictedgenreaccuracy.adjustSize()

def window():
    app = QApplication(sys.argv)
    win = MyWindow()

    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    window()


