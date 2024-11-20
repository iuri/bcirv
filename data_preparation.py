import os
import re
import mne
from mne.io import read_raw_edf, concatenate_raws

from scipy.signal import iirnotch, filtfilt
import numpy as np




def load_files(root_directory='./datasets', subjects_n=1, ref_sfreq=160):
  # Loads edf files into a list
  edf_files = []
  for subject_folder in os.listdir(root_directory):
      # collect only samples from runs 3,4,7,8,11,12 of 30 subjects. (i.e. 6*30=180)
      # 9999 to collect data from all 110 subjects
      # to adjust the loop to one subject only, replace 480 with 6
      if len(edf_files) == subjects_n*6:
          break
      subject_path = os.path.join(root_directory, subject_folder)
      if os.path.isdir(subject_path):
          subject_edf_files = [
              os.path.join(subject_path, file)
              for file in os.listdir(subject_path)
              # filter left and right firsts trials
              if file.endswith(".edf") and len(re.findall('S.*R03|R04|R07|R08|R11|R12', file)) > 0
          ]
          # check the sampling rate if the first file
          if subject_edf_files:
              # ref_sfreq = read_raw_edf(subject_edf_files[0], preload=False).info['sfreq']
              print('Sampling rate', ref_sfreq)
              # check the sampling rate of the rest of the files
              consistent_sampling_rate = all(
                  int(read_raw_edf(file, preload=False).info['sfreq']) == int(ref_sfreq)
                  for file in subject_edf_files[1:]
              )
              if consistent_sampling_rate:
                  edf_files.extend(subject_edf_files)
              else:
                  print(f"Skipping subject {subject_folder} due to inconsistent sampling rates.")


def load_data(edf_files, low_freq=7.5, high_freq=30):
  # Load and prepare raw data from EDF files 
  data = []
  for a in range(1,len(edf_files),6):
      # print("a: ", a)
      # filter 3 trials for a single subject from edf_files list
      filtered_files = [edf_files[i] for i in [a,a+2,a+4]]
      # print(filtered_files)
      # load trials for a single subject only
      raw_files = [read_raw_edf(f, preload=True) for f in filtered_files]
      # concatenate 3 trials in 1 single raw data
      raw = concatenate_raws(raw_files, preload=True)

      # parse channels
      prep = parse_channels(raw)

      # apply band pass filter to raw data
      prep1 = prep.filter(l_freq=low_freq, h_freq=high_freq, fir_design='firwin')

      prep2 = apply_ica(prep1)

      # apply notch filter, eliminate 60Hz standard baseline eletric voltage
      prep3 = apply_notch_filter(prep2)

      # Apply baseline correction using median
      # mne.baseline.rescale(prep3, times=prep1.times, baseline=baseline, mode='logratio')
      # prep4 = mne.baseline.rescale(prep3.get_data(), times=prep1.times, baseline=baseline, mode='logratio')

      # append raw for each subject to the list raws
      data.append(prep3)

  return data

def apply_ica(data):
  # Remove T9 and T10 because of errors with ICA
  data.drop_channels(['T9', 'T10'])
  # Apply Independent Component Analysis (ICA) to identify and remove artifacts
  ica = mne.preprocessing.ICA(n_components=20, method='fastica', random_state=97, max_iter=1000)
  ica.fit(data)
  # Plot ICA components and properties
  ## ica.plot_components()
  ## ica.plot_properties(data, picks=[0, 1, 2])  # Inspect the first few components
  # Identify and Remove Artifact Components
  ## ica.exclude = [0, 1]  # Example: exclude components 0 and 1 identified as artifacts
  # Apply ICA to the data
  return ica.apply(data.copy())

def notch_filter(data, fs, freq, quality_factor):
    """
    Apply a notch filter to the EEG data.

    Parameters:
    data (numpy.ndarray): The EEG data, shape (n_channels, n_samples).
    fs (float): Sampling frequency of the data.
    freq (float): Frequency to be removed (e.g., 50 or 60 Hz).
    quality_factor (float): Quality factor for the notch filter.

    Returns:
    numpy.ndarray: The filtered EEG data.
    """
    # Design the notch filter
    b, a = iirnotch(w0=freq, Q=quality_factor, fs=fs)

    # Apply the notch filter to each channel
    filtered_data = np.zeros_like(data)
    for channel in range(data.shape[0]):
        filtered_data[channel, :] = filtfilt(b, a, data[channel, :])

    return filtered_data

def apply_notch_filter(data):
  fs = data.info['sfreq']  # Sampling frequency
  n_channels = data.get_data().shape[0]
  n_samples = data.get_data().shape[1]
  eeg_data = data.get_data()

  # Add a 50 Hz sinusoidal noise to the data for demonstration
  # t = np.arange(n_samples) / fs
  noise_freq = 50  # Noise frequency
  # eeg_data += 0.5 * np.sin(2 * np.pi * noise_freq * t)

  # Apply the notch filter
  quality_factor = 100  # Example quality factor
  data._data = notch_filter(eeg_data, fs, noise_freq, quality_factor)
  return data



def parse_channels(data):
  # rename channels to standard
  # Define the mapping between old and new channel names
  channel_mapping = {'C3..': 'C3', 'Cz..': 'Cz', 'C4..': 'C4',
                    'Fp1.': 'Fp1', 'Fp2.': 'Fp2', 'Fpz.': 'Fpz',
                    'Fc5.': 'FC5', 'Fc3.':'FC3', 'Fc1.':'FC1',
                    'Fcz.':'FCz', 'Fc2.':'FC2', 'Fc4.':'FC4', 'Fc6.':'FC6',
                    'C5..':'C5', 'C1..':'C1', 'C2..':'C2', 'C6..':'C6',
                    'Cp5.':'CP5', 'Cp3.':'CP3', 'Cp1.':'CP1', 'Cpz.':'CPz',
                    'Cp2.':'CP2', 'Cp4.':'CP4', 'Cp6.':'CP6', 'Af7.':'AF7',
                    'Af3.':'AF3', 'Afz.':'AFz', 'Af4.':'AF4', 'Af8.':'AF8',
                    'F7..':'F7', 'F5..':'F5', 'F3..':'F3', 'F1..':'F1',
                    'Fz..':'Fz', 'F2..':'F2', 'F4..':'F4', 'F6..':'F6',
                    'F8..':'F8', 'Ft7.':'FT7', 'Ft8.':'FT8', 'T7..':'T7',
                    'T8..':'T8', 'T9..':'T9', 'T10.':'T10', 'Tp7.':'TP7',
                    'Tp8.':'TP8', 'P7..':'P7', 'P5..':'P5', 'P3..':'P3',
                    'P1..':'P1', 'Pz..':'Pz', 'P2..':'P2', 'P4..':'P4',
                    'P6..':'P6', 'P8..':'P8', 'Po7.':'PO7', 'Po3.':'PO3',
                    'Poz.':'POz', 'Po4.':'PO4', 'Po8.':'PO8', 'O1..':'O1',
                    'Oz..':'Oz', 'O2..':'O2', 'Iz..':'Iz'}
  data.rename_channels(channel_mapping)
  # Set montage (standard_1005)
  # montage = mne.channels.make_standard_montage('easycap-M1')
  # montage = mne.channels.make_standard_montage('easycap-M2')
  # montage = mne.channels.make_standard_montage('standard_1005')
  # montage = mne.channels.make_standard_montage('standard_1020')
  montage = mne.channels.make_standard_montage('biosemi64')
  print(len(montage.ch_names),montage.ch_names)
  # print(mne.io.constants.CHANNEL_LOC_ALIASES)
  print(montage)

  data.set_montage(montage, match_case=False, match_alias=True, on_missing='ignore')
  # raw3.set_montage('easycap-M1')
  # raw3.plot_sensors()

  return data



def prepare_dataset(raw, low_freq=7.5, high_freq=30, channels=['Fp1', 'Fpz', 'Fp2']):
  # Select channels
  prep = parse_channels(raw)

  # apply band pass filter to raw data
  prep1 = prep.filter(l_freq=low_freq, h_freq=high_freq, fir_design='firwin')

  prep2 = apply_ica(prep1)

  # apply notch filter, eliminate 60Hz standard baseline eletric voltage
  prep3 = apply_notch_filter(prep2)
  # print(prep3.info)

  prep4 = prep3.pick_channels(channels)

  return prep4