import mne
import numpy as np
from scipy.signal import welch, hilbert
from scipy.stats import skew, kurtosis

import pandas as pd

from PyEMD import EMD
import pywt

## Power Spectral Density
def compute_psd(data, fs):
    n_epochs, n_channels, n_samples = data.shape
    psd_all_epochs = []
    freqs_all_epochs = []

    for epoch in range(n_epochs):
        psd_epoch = []
        freqs_epoch = []
        for channel in range(n_channels):
            freqs, psd = welch(data[epoch, channel, :], fs)
            psd_epoch.append(psd)
            freqs_epoch.append(freqs)
        psd_all_epochs.append(psd_epoch)
        freqs_all_epochs.append(freqs_epoch)

    psd_all_epochs = np.array(psd_all_epochs)  # Shape: [n_epochs, n_channels, n_freqs]
    freqs_all_epochs = np.array(freqs_all_epochs)  # Shape: [n_epochs, n_channels, n_freqs]
    return freqs_all_epochs, psd_all_epochs




# Band Power
def band_power(freqs, psd, band):
    low, high = band
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    band_power = np.trapz(psd[idx_band], freqs[idx_band])
    return band_power

def compute_band_powers(freqs_all_epochs, psd_all_epochs, frequency_bands):
    n_epochs, n_channels, n_freqs = psd_all_epochs.shape
    band_powers = {band: np.zeros((n_epochs, n_channels)) for band in frequency_bands}
    for epoch in range(n_epochs):
        for channel in range(n_channels):
            for band in frequency_bands:
              band_powers[band][epoch, channel] = band_power(freqs_all_epochs[epoch, channel], psd_all_epochs[epoch, channel], frequency_bands[band])

    return band_powers



## Wavelet Coefficients

def calculate_wavelet_coefficients(data, wavelet='db4', level=4):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def calculate_wavelet_coeffs_all_epochs(data, wavelet='db4', level=4):
    n_epochs, n_channels, n_samples = data.shape
    all_coeffs = []

    for epoch in range(n_epochs):
        epoch_coeffs = []
        for channel in range(n_channels):
            coeffs = calculate_wavelet_coefficients(data[epoch, channel, :], wavelet, level)
            epoch_coeffs.append(coeffs)
        all_coeffs.append(epoch_coeffs)

    return all_coeffs




# Function to pad wavelet coefficients to ensure uniform shape
def pad_wavelet_coeffs(wavelet_coeffs):
    max_length = max(max(len(coeff) for coeff in channel) for epoch in wavelet_coeffs for channel in epoch)

    padded_coeffs = []
    for epoch in wavelet_coeffs:
        epoch_coeffs = []
        for channel in epoch:
            channel_coeffs = []
            for coeff in channel:
                padded_coeff = np.pad(coeff,
                                      (0, max_length - len(coeff)),
                                      mode='constant', constant_values=0)
                channel_coeffs.append(padded_coeff)
            epoch_coeffs.append(channel_coeffs)
        padded_coeffs.append(epoch_coeffs)
    return np.array(padded_coeffs)






## Hilbert-Huang Transform (HHT)

# Function to compute EMD and extract Hilbert features for one signal
def compute_hilbert_features(signal):
    # Perform EMD
    emd = EMD()
    imfs = emd.emd(signal)
    num_imfs = imfs.shape[0]

    # Calculate instantaneous frequency and amplitude for each IMF
    instantaneous_frequencies = []
    instantaneous_amplitudes = []

    for imf in imfs:
        analytic_signal = hilbert(imf)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)  # In cycles per sample

        instantaneous_frequencies.append(instantaneous_frequency)
        instantaneous_amplitudes.append(amplitude_envelope[:-1])  # Match lengths with frequency

    instantaneous_frequencies = np.array(instantaneous_frequencies)
    instantaneous_amplitudes = np.array(instantaneous_amplitudes)

    # Function to compute energy
    def compute_energy(signal):
        return np.sum(np.square(signal))

    # Function to compute entropy
    def compute_entropy(signal):
        power_spectrum = np.square(signal)
        ps_norm = power_spectrum / np.sum(power_spectrum)
        return -np.sum(ps_norm * np.log2(ps_norm + 1e-10))

    # Extract features
    features = []
    for i in range(num_imfs):
        mean_freq = np.mean(instantaneous_frequencies[i])
        std_freq = np.std(instantaneous_frequencies[i])
        mean_amp = np.mean(instantaneous_amplitudes[i])
        std_amp = np.std(instantaneous_amplitudes[i])
        energy_amp = compute_energy(instantaneous_amplitudes[i])
        entropy_amp = compute_entropy(instantaneous_amplitudes[i])

        features.extend([mean_freq, std_freq, mean_amp, std_amp, energy_amp, entropy_amp])

    return np.array(features)


def extract_features(prep, epochs, channels=['Fp1', 'Fpz', 'Fp2'], tmin=-0.2, tmax=1.0 ):
    # Compute PSD for all epochs
    freqs_all_epochs, psd_all_epochs = compute_psd(epochs.get_data(), prep.info['sfreq'])

    # Band Power
    frequency_bands = {
        'mi': (7.5, 12.5),
        'beta': (12, 30),
    }
    # Compute band powers for all epochs and channels
    band_powers = compute_band_powers(freqs_all_epochs, psd_all_epochs, frequency_bands)

    # Compute wavelet coefficients for all epochs
    all_wavelet_coeffs = calculate_wavelet_coeffs_all_epochs(epochs.get_data())

    
    # Example EEG data with the provided shape
    epochs_data = epochs.get_data()

    # Apply the function to each channel of each epoch
    hh_all_features = []
    for epoch in epochs_data:
        for channel in epoch:
            features = compute_hilbert_features(channel)
            hh_all_features.append(features)

    # Convert to NumPy array for further processing (e.g., ML models)
    hh_all_features = np.array(hh_all_features)

    # print("Hilbert-Huang Features shape:", hh_all_features)


    # Combine the 4 features: PSD, Band powers, Wavelet Coefficients and Hilbert Huang Coefficients, to serve as input of CNN training model
    psd_features = psd_all_epochs
    # print("PSD Features shape:", psd_features.shape)  # Should be (87, 62, 2)


    # Convert band power features dictionary to numpy arrays
    mi_band_power = band_powers['mi'].reshape(band_powers['mi'].shape[0], band_powers['mi'].shape[1], 1)
    beta_band_power = band_powers['beta'].reshape(band_powers['beta'].shape[0], band_powers['beta'].shape[1], 1)
    # Concatenate along the feature dimension
    band_power_features = np.concatenate((mi_band_power, beta_band_power), axis=2)
    # print("Band Power Features shape:", band_power_features.shape)  # Should be (87, 62, 2)


    # Example of wavelet coefficients structure for demonstration
    # Let's create a dummy wavelet coefficients structure with the specified shape
    # np.random.seed(0)
    # all_wavelet_coeffs = [[[[np.random.rand(18) for _ in range(5)] for _ in range(62)] for _ in range(87)]]
    # Pad the wavelet coefficients to ensure uniform length
    padded_wavelet_coeffs = pad_wavelet_coeffs(all_wavelet_coeffs)

    # Verify the shape
    # print("Padded Wavelet Coefficients shape:", padded_wavelet_coeffs.shape)  # Should be (45, 62, 5, max_length)

    # Flatten the last two dimensions to combine the coefficients and samples into one dimension
    wavelet_coeffs_flattened = padded_wavelet_coeffs.reshape(padded_wavelet_coeffs.shape[0], padded_wavelet_coeffs.shape[1], -1)
    # print("Flattened Wavelet Coefficients shape:", wavelet_coeffs_flattened.shape)  # Should be (45, 62, 5*max_length)



    # Reshape Hilbert-Huang features to (45, 3, 12)
    hilbert_huang_features = hh_all_features.reshape(int(hh_all_features.shape[0]/len(channels)), len(channels), hh_all_features.shape[1])
    # print("Hilbert-Huang Features shape:", hilbert_huang_features.shape)  # Should be (45, 3, 12)


    # Concatenate PSD, Band Powers, Wavelet Coefficients, and Hilbert-Huang features along the last axis
    frequency_features = np.concatenate((psd_features, band_power_features, wavelet_coeffs_flattened, hilbert_huang_features), axis=2)
    # print("Combined Features shape:", frequency_features.shape)  # Should be (87, 62, total_features)



    ##
    # Time Domain Features
    ##

    # Load and combine data for all subjects, segmented by epochs left, right and relax
    all_data = []
    df_combined = pd.DataFrame()
    # Parameters for epoch extraction
    tmin, tmax = -0.2, 1.0 # interval window to collect data 0.2 seconds before the event 1 second after
   
    # events, event_ids = mne.events_from_annotations(r)
    events, _ = mne.events_from_annotations(prep)
    event_ids={'left':2,'right':3}
    epochs = mne.Epochs(prep, events, event_id=event_ids, tmin=tmin, tmax=tmax, preload=True)
    labels = epochs.events[:, -1]
    subject_data = epochs.get_data(picks=channels).mean(axis=2)

    for j, task in enumerate(event_ids.keys()):
        task_data = subject_data[labels == event_ids[task]]
        # print('TASK shape:', task_data.shape)
        task_labels = np.full(task_data.shape[0], task)
        df = pd.DataFrame(task_data, columns=channels)
        df['Task'] = task_labels
        df_combined = pd.concat([df_combined, df],ignore_index=True)
        all_data.append(df)

    # filter left T1 and right T2 values only, now as 1 and 0
    df_filtered = df_combined[df_combined['Task'] != 'T0'].replace({'left':0, 'right':1})
    # df_filtered

    df_channels  = df_filtered.iloc[:,:len(channels)]
    # df_channels

    # Initialize lists to store results
    medians = []
    means = []
    variances = []
    skewnesses = []
    kurtoses = []
    std_devs = []

    # Calculate metrics for each row
    for index, row in df_channels.iterrows():
        medians.append(np.median(row))
        means.append(np.mean(row))
        variances.append(np.var(row))
        skewnesses.append(skew(row))
        kurtoses.append(kurtosis(row))
        # Calculate Standard Deviantion
        std_devs.append(np.std(row))


    # Create a DataFrame to display the results
    results_df = pd.DataFrame({
        'Median': medians,
        'Mean': means,
        'Variance': variances,
        'Skewness': skewnesses,
        'Kurtosis': kurtoses,
        'Standard Deviation': std_devs
    })

    df1 = df_channels
    df1 = df1.reset_index(drop=True)
    df1 = pd.concat([df1, results_df],axis=1)
    # df1.head(4)


    # Combine Frequency and Time features

    time_features = np.array(df1.values)
    # print("Time Domain Features Shape", time_features.shape)
    # print("Frequency Domain Features Shape", frequency_features.shape)


    # Assuming time_features has fewer samples and can be duplicated for simplicity
    time_features_reshaped = np.tile(time_features[:, np.newaxis, :], (1, frequency_features.shape[1], 1))
    # print("Time Domain Features Reshaped", time_features_reshaped.shape)

    # Concatenate along the last axis
    combined_features = np.concatenate([time_features_reshaped, frequency_features], axis=-1)
    # print("Combined Features Shape:", combined_features.shape)
    # print(combined_features.shape, combined_features)


    return combined_features