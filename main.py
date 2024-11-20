import os
import joblib
import argparse
import numpy as np 

import mne 
from mne.io import read_raw_edf

from data_preparation import prepare_dataset
from feature_extraction import extract_features
from cnn import select_features_cnn1D

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from utils import write_output




def main(filename):
    ref_sfreq = 160
    low_freq = 7.5
    high_freq = 30.0
    # channels = ['Fp1', 'Fpz', 'Fp2', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4']
    channels = ['Fp1', 'Fpz', 'Fp2']
   
    # Load EDF file
    raw = read_raw_edf(os.path.join('./datasets', filename), preload=True)
    if int(raw.info['sfreq']) != int(ref_sfreq):
        print(f"Sampling Frequency {int(ref_sfreq)}")
        return 
    
    # Prepare dataset
    prep = prepare_dataset(raw, low_freq, high_freq, channels)

   
    # Extract Events
    events, event_ids = mne.events_from_annotations(prep)

    # event_ids={'rest':1,'left':2,'right':3}
    event_ids={'left':2,'right':3}
    # plot events (i.e. left and right motor movements) for a each trial separatedely of a subject
    # fig=mne.viz.plot_events(events,event_id=event_ids,sfreq=prep4.info['sfreq'],first_samp=prep4.first_samp)

    # Get Epochs
    # extract the time interval when the events left and right happen
    tmin, tmax = -0.2, 1.0 # interval window to collect data 0.2 seconds before the event 1 second after
    epochs = mne.Epochs(prep, events, event_id=event_ids, tmin=tmin, tmax=tmax, preload=True)


    ###
    # Feature Extraction
    ### 
    features = extract_features(prep, epochs, channels, tmin, tmax)



    labels = epochs.events[:, -1]
    # Replace 2 with 0 and 3 with 1
    labels = np.where(labels == 2, 0, np.where(labels == 3, 1, labels))
    # print(labels.shape,labels)

    # features = select_features_cnn1D(features, labels)
    ##
    # Classification / Modeling
    ###
    features_reshaped = features.reshape(features.shape[0], features.shape[1]*features.shape[2])

    # Load the model for future predictions
    model_filename = "./models/random_forest_classifier_full.joblib"
    rf_classifier = joblib.load(model_filename)
    X_test_flat = features_reshaped.reshape(features_reshaped.shape[0], -1)
    y_pred_rf = rf_classifier.predict(X_test_flat)




    # Normalize and scale data
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(features_reshaped)

    # Load the model for future predictions
    model_filename = "./models/random_forest_classifier_full.joblib"
    rf_classifier = joblib.load(model_filename)
    X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
    y_pred_rf = rf_classifier.predict(X_test_flat)

   
    write_output(events, y_pred_rf)
    # Evaluate the combined model
    accuracy_combined = accuracy_score(labels, y_pred_rf)
    print(f'Random Forest Model Accuracy: {accuracy_combined:.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='bciclassifier', description="BCI classifier")
    parser.add_argument(
        '--filename',
        type=str,
        help="EDF filename within current directory",
        default=None
    )
    parser.print_help()
   
    args = parser.parse_args()

    main(args.filename)
