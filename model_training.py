
import joblib

from data_preparation import * 
from feature_extraction import *

import pandas as pd

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score

from cnn import select_features

from classification import xgb_classify




def main():
    subjects_n = 1
    ref_sfreq = 160
    low_freq = 7.5
    high_freq = 30.0
    # Define the baseline period (e.g., from -0.5 to 0 seconds)
    baseline = (-0.2, 0)

    # channels = ['Fp1', 'Fpz', 'Fp2', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4']
    channels = ['Fp1', 'Fpz', 'Fp2']

    root_directory = './datasets'
   

    edf_files = load_files(root_directory, subjects_n, ref_sfreq)

    data = load_data(edf_files, low_freq, high_freq)
  

   
    prep4 = data[0].pick_channels(channels)

    # Extract Events
    events, event_ids = mne.events_from_annotations(prep4)

    # event_ids={'rest':1,'left':2,'right':3}
    event_ids={'left':2,'right':3}
    # plot events (i.e. left and right motor movements) for a each trial separatedely of a subject
    # fig=mne.viz.plot_events(events,event_id=event_ids,sfreq=prep4.info['sfreq'],first_samp=prep4.first_samp)

    # exit()
    # Get Epochs
    # extract the time interval when the events left and right happen
    tmin, tmax = -0.2, 1.0 # interval window to collect data 0.2 seconds before the event 1 second after
    epochs = mne.Epochs(prep4, events, event_id=event_ids, tmin=tmin, tmax=tmax, preload=True)


    ###
    # Feature Extraction
    ### 
    features = extract_features(prep, epochs, channels, tmin, tmax)


    features = select_features(features)

    labels = epochs.events[:, -1]
    # Replace 2 with 0 and 3 with 1
    labels = np.where(labels == 2, 0, np.where(labels == 3, 1, labels))
    # print(labels.shape,labels)



    ##
    # Classification / Modeling
    ###
    features_reshaped = features.reshape(features.shape[0], features.shape[1]*features.shape[2])


    # Split data into test and train samples
    X_train, X_test, y_train, y_test = train_test_split(features_reshaped, labels, test_size=0.3, random_state=42)

    # Normalize and scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Random Forest classification
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred_rf = rf_classifier.predict(X_test_flat)


    # Save the model using joblib
    model_filename = "random_forest_classifier.joblib"
    joblib.dump(rf_classifier, model_filename)
    print(f"Model saved as {model_filename}")




    # Evaluate the combined model
    accuracy_combined = accuracy_score(y_test, y_pred_rf)
    print(f'Random Forest Model Accuracy: {accuracy_combined:.2f}')

    print('Classification Report:')
    print(classification_report(y_test, y_pred_rf))


    ##
    # XG Boost Classifier
    ##
    y_pred = xgb_classify(X_train, X_test, y_train)

    # Step 6: Evaluate Model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)


    y_pred = xgb_classify(X_train_scaled, X_test_scaled, y_train)

    # Step 6: Evaluate Model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multiclass
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

  
if __name__ == "__main__":
    main()
