import os
import sys
import pandas as pd
import pyeeg
from collections import OrderedDict
from catboost import CatBoostClassifier
import pickle as pkl
import statistics
import scipy.spatial as ss
import scipy.stats as sst
from PyQt4 import QtCore, QtGui, Qt
import numpy as np
from modbase import ModuleBase
import time

class EmotionRecognition:
    def __init__(self, path="data_preprocessed_python/", format='dat'):
        def first_der(x):
            return np.mean(np.abs(x[1:] - x[0:-1]))

        def second_der(x):
            return np.mean(np.abs(x[2:] - x[0:-2]))

        def power_4_8(X):
            return pyeeg.bin_power((X * 1e-3) ** 2, [4, 8], 512)[0][0]

        def power_8_12(X):
            return pyeeg.bin_power((X * 1e-3) ** 2, [8, 12], 500)[0][0]

        def power_12_25(X):
            return pyeeg.bin_power((X * 1e-3) ** 2, [12, 25], 500)[0][0]

        def power_35_45(X):
            return pyeeg.bin_power((X * 1e-3) ** 2, [35, 45], 500)[0][0]



        def freq_std_theta(X, Fs=500):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            theta = frequency[L * 4 // Fs - 1: L * 8 // Fs]
            return np.std(theta)

        def freq_mean_theta(X, Fs=500):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            theta = frequency[L * 4 // Fs - 1: L * 8 // Fs]
            return np.mean(theta)

        def freq_std_alpha(X, Fs=500):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            alpha = frequency[L * 5 // Fs - 1: L * 13 // Fs]
            return np.std(alpha)

        def freq_mean_alpha(X, Fs=500):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            alpha = frequency[L * 5 // Fs - 1: L * 13 // Fs]
            return np.mean(alpha)

        def freq_std_beta(X, Fs=500):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            beta = frequency[L * 13 // Fs - 1: L * 30 // Fs]
            return np.std(beta)

        def freq_mean_beta(X, Fs=500):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            beta = frequency[L * 13 // Fs - 1: L * 30 // Fs]
            return np.mean(beta)

        def freq_std_gamma(X, Fs=500):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            gamma = frequency[L * 30 // Fs - 1: L * 50 // Fs]
            return np.std(gamma)

        def freq_mean_gamma(X, Fs=500):
            L = len(X)
            data_fft = np.fft.fft(X)
            frequency = abs(data_fft / L)
            frequency = frequency[: L // 2 + 1] * 2
            gamma = frequency[L * 30 // Fs - 1: L * 50 // Fs]
            return np.mean(gamma)

        self.functions = [np.mean,
             np.std,first_der,second_der, statistics.median, power_4_8,
             power_8_12, power_12_25, power_35_45,
             freq_std_theta, freq_std_alpha,
             freq_std_beta, freq_std_gamma, freq_mean_theta, freq_mean_alpha,
             freq_mean_beta, freq_mean_gamma]

    def make_features(self, data):
        features_all = []
        good_chanels = [1, 2, 3, 4, 7, 11, 13, 31, 29, 27, 21, 20, 19, 17]
        ch_names = ["AF3", "F3", "F7", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F8", "F4", "AF4"]
        features = dict()
        for k in range(14):
            for func in self.functions:
                features["channel_" + ch_names[k] + "_" + func.__name__] = func(data[good_chanels[k]])
        features_all.append(features)
        return pd.DataFrame(features_all)

    def predict_measure(self, data, model):
        return model.predict_proba(data)

    def predict_emotion(self, feature, X_train, y_train_ar, y_train_va):
        """
        Get arousal and valence class from feature.
        Input: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
        Output: Class of emotion between 1 to 3 from each arousal and valence. 1 denotes low category, 2 denotes normal category, and 3 denotes high category.
        """
        # Compute canberra with arousal training data
        distance_ar = np.array(list(map(lambda x: ss.distance.canberra(x, feature), np.array(X_train))))

        # Compute canberra with valence training data
        distance_va = np.array(list(map(lambda x: ss.distance.canberra(x, feature), np.array(X_train))))

        # Compute 3 nearest index and distance value from arousal
        idx_nearest_ar = np.array(np.argsort(distance_ar)[:3])
        val_nearest_ar = np.array(np.sort(distance_ar)[:3])
        # Compute 3 nearest index and distance value from arousal
        idx_nearest_va = np.array(np.argsort(distance_va)[:3])
        val_nearest_va = np.array(np.sort(distance_va)[:3])

        # Compute comparation from first nearest and second nearest distance. If comparation less or equal than 0.7, then take class from the first nearest distance. Else take frequently class.
        # Arousal
        comp_ar = val_nearest_ar[0] / val_nearest_ar[1]
        if comp_ar <= 0.97:
            result_ar = np.array(y_train_ar)[idx_nearest_ar[0]]
        else:
            result_ar = sst.mode(np.array(y_train_ar)[idx_nearest_ar])
            result_ar = float(result_ar[0])

        # Valence
        comp_va = val_nearest_va[0] / val_nearest_va[1]
        if comp_va <= 0.97:
            result_va = np.array(y_train_va)[idx_nearest_va[0]]
        else:
            result_va = sst.mode(np.array(y_train_va)[idx_nearest_va])
            result_va = float(result_va[0])
        return result_ar, result_va

    def determine_emotion_class(self, class_ar, class_va):
        if class_ar == 2.0 or class_va == 2.0:
            emotion_class = 5
        elif class_ar == 3.0 and class_va == 1.0:
            emotion_class = 1
        elif class_ar == 3.0 and class_va == 3.0:
            emotion_class = 2
        elif class_ar == 1.0 and class_va == 3.0:
            emotion_class = 3
        elif class_ar == 1.0 and class_va == 1.0:
            emotion_class = 4

        return emotion_class
    def add_abs_power(self, data_all):
        ch_names = ["AF3", "F3", "F7", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F8", "F4", "AF4"]
        for k in range(14):
            sum_pow = data_all["channel_" + str(ch_names[k]) + "_power_4_8"] + data_all["channel_" + str(ch_names[k]) + "_power_8_12"] + data_all["channel_" + str(ch_names[k]) + "_power_12_25"] + data_all["channel_" + str(ch_names[k]) + "_power_35_45"]

            data_all["channel_" + str(ch_names[k]) + "_power_4_8_abs"] = data_all["channel_" + str(ch_names[k]) + "_power_4_8"] / sum_pow
            data_all["channel_" + str(ch_names[k]) + "_power_8_12_abs"] = data_all["channel_" + str(ch_names[k]) + "_power_8_12"] / sum_pow
            data_all["channel_" + str(ch_names[k]) + "_power_12_25_abs"] = data_all["channel_" + str(ch_names[k]) + "_power_12_25"] / sum_pow
            data_all["channel_" + str(ch_names[k]) + "_power_35_45_abs"] = data_all["channel_" + str(ch_names[k]) + "_power_35_45"] / sum_pow
        od = OrderedDict(sorted(data_all.items(), key=lambda t: t[0]))
        return pd.DataFrame(od)

class EMO(ModuleBase):
    ''' Tutorial Module 0.
    Minimum implementation for a module
    '''

    def __init__(self, *args, **keys):
        ''' Constructor
        '''
        # initialize the base class, give a descriptive name
        ModuleBase.__init__(self, name="Tutorial 0", **keys)


        # initialize module variables
        self.data = None  # hold the data block we got from previous module
        self.dataavailable = False  # data available for output to next module
        self.data_queue = []
        self.ended = []
        self.rec = EmotionRecognition()
        with open('cat_valence_model.pkl', 'rb') as f:
            self.model_valence = pkl.load(f)
        with open('cat_arousal_model.pkl', 'rb') as f:
            self.model_arousal = pkl.load(f)
        self.X_train = pd.read_csv('X_trainer.csv')
        self.y_train_ar = pd.read_csv('y_train_ar_last.csv')
        self.y_train_va = pd.read_csv('y_train_va_last.csv')

        self.onlinePane = _OnlineCfgPane()

        # connect the signal handler for trigger out settings
        #self.connect(self.online_cfg, Qt.SIGNAL("valueChanged(int)"), self.sendTrigger)
    def setDefault(self):
        ''' Set all module parameters to default values
        '''
        self.chunk = 1024
        self.frequency_range = 200
        self.plot_items = 4
        self.onlinePane.setCurrentValues(self.frequency_range, self.chunk)
    def determine_emotion(self, arousal, valence):
        if arousal == 0 and valence == 0:
            return "Bored/Tired/Depressed"
        if arousal == 1 and valence == 0:
            return "Sad/Miserable/Unhappy"
        if arousal == 0 and valence == 1:
            return "Sleepy/Still/Quiet"
        if arousal == 1 and valence == 1:
            return "Neutral"
        if arousal == 0 and valence == 2:
            return "Calm/Relaxed"
        if arousal == 2 and valence == 0:
            return "Tense/Nervous/Stressed"
        if arousal == 1 and valence == 2:
            return "Happy/Pleased"
        if arousal == 2 and valence == 1:
            return "Aroused/Hyper-activated"
        if arousal == 2 and valence == 2:
            return "Excited/Elated"

    def get_online_configuration(self):
        ''' Get the online configuration pane
        @return: a QFrame object or None if you don't need a online configuration pane
        '''
        return self.onlinePane

    def process_input(self, datablock):
        ''' Get data from previous module
        @param datablock: EEG_DataBlock object
        '''
        self.dataavailable = True  # signal data availability
        self.data = datablock  # get a local reference
        self.data_queue.append(datablock.eeg_channels)
        self.ended.append(datablock.eeg_channels)
        if len(self.data_queue) > 200:
            raw_data = np.hstack(self.data_queue)
            features = self.rec.add_abs_power(self.rec.make_features(raw_data))
            valence_pred = self.model_valence.predict(features)[0][0]
            arousal_pred = self.model_arousal.predict(features)[0][0]
            pred_knn = self.rec.predict_emotion(features, self.X_train, self.y_train_ar, self.y_train_va)
            arousal = pred_knn[0]
            valence = pred_knn[1]
            self.onlinePane.labelFrequency.setText(self.determine_emotion(arousal_pred, valence_pred) +
                                                   '\n' + 'arousal: ' + str('low' if arousal_pred == 0 else 'medium' if
                                                    arousal_pred == 1 else 'high') +
                                                   '\n' + 'valence: ' + str('low' if valence_pred == 0 else 'medium'
                                                    if valence_pred == 1 else 'high'))
            self.onlinePane.knn.setText(self.determine_emotion(arousal, valence) +
                                                   '\n' + 'arousal: ' + str('low' if arousal == 0 else 'medium' if
                                                    arousal == 1 else 'high') +
                                                   '\n' + 'valence: ' + str('low' if valence == 0 else 'medium'
                                                    if valence == 1 else 'high'))

            with open("pred_log.txt", "a+") as f:
                f.write(time.ctime() + ": " + self.determine_emotion(arousal, valence) +
                                                   ', ' + 'arousal: ' + str('low' if arousal == 0 else 'medium' if
                                                    arousal == 1 else 'high') +
                                                   ', ' + 'valence: ' + str('low' if valence == 0 else 'medium'
                                                    if valence == 1 else 'high') + '\n')
            f.close()
            for i in range(100):
                self.data_queue.pop(0)
    def process_output(self):
        ''' Send data out to next module
        '''
        if not self.dataavailable:
            return None
        self.dataavailable = False
        return self.data


class _OnlineCfgPane(Qt.QFrame):
    ''' Online configuration pane
    '''

    def __init__(self, *args):
        apply(Qt.QFrame.__init__, (self,) + args)

        # make it nice ;-)
        self.setFrameShape(QtGui.QFrame.Panel)
        self.setFrameShadow(QtGui.QFrame.Raised)

        # give us a layout and group box
        self.gridLayout = QtGui.QGridLayout(self)
        self.groupBox = QtGui.QGroupBox(self)
        self.groupBox.setTitle("Emotion Recognition")
        # group box layout
        self.gridLayoutGroup = QtGui.QGridLayout(self.groupBox)
        self.gridLayoutGroup.setHorizontalSpacing(10)
        self.gridLayoutGroup.setContentsMargins(20, -1, 20, -1)

        # add the chunk size combobox
        '''
        self.comboBoxChunk = QtGui.QComboBox(self.groupBox)
        self.comboBoxChunk.setObjectName("comboBoxChunk")
        self.comboBoxChunk.addItem(Qt.QString("128"))
        self.comboBoxChunk.addItem(Qt.QString("256"))
        self.comboBoxChunk.addItem(Qt.QString("512"))
        self.comboBoxChunk.addItem(Qt.QString("1024"))
        self.comboBoxChunk.addItem(Qt.QString("2048"))
        self.comboBoxChunk.addItem(Qt.QString("4096"))
        self.comboBoxChunk.addItem(Qt.QString("8129"))
        self.comboBoxChunk.addItem(Qt.QString("16384"))
        self.comboBoxChunk.addItem(Qt.QString("32768"))
        '''
        # add the frequency range combobox
        '''
        self.comboBoxFrequency = QtGui.QComboBox(self.groupBox)
        self.comboBoxFrequency.setObjectName("comboBoxChunk")
        self.comboBoxFrequency.addItem(Qt.QString("20"))
        self.comboBoxFrequency.addItem(Qt.QString("50"))
        self.comboBoxFrequency.addItem(Qt.QString("100"))
        self.comboBoxFrequency.addItem(Qt.QString("200"))
        self.comboBoxFrequency.addItem(Qt.QString("500"))
        self.comboBoxFrequency.addItem(Qt.QString("1000"))
        self.comboBoxFrequency.addItem(Qt.QString("2000"))
        self.comboBoxFrequency.addItem(Qt.QString("5000"))
        '''
        # create unit labels
        self.labelChunk = QtGui.QLabel(self.groupBox)
        #self.labelChunk.setText("[n]")
        self.labelFrequency = QtGui.QLabel(self.groupBox)
        self.knn = QtGui.QLabel(self.groupBox)
        #self.labelFrequency.setText("[Hz]")

        # add widgets to layouts
        #self.gridLayoutGroup.addWidget(self.comboBoxFrequency, 0, 0, 1, 1)
        self.gridLayoutGroup.addWidget(self.labelFrequency, 0, 1, 1, 1)
        self.gridLayoutGroup.addWidget(self.knn, 0, 2, 1, 1)
        #self.gridLayoutGroup.addWidget(self.comboBoxChunk, 0, 2, 1, 1)
        self.gridLayoutGroup.addWidget(self.labelChunk, 0, 3, 1, 1)

        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)

        # set default values
        #self.comboBoxFrequency.setCurrentIndex(2)
        #self.comboBoxChunk.setCurrentIndex(4)