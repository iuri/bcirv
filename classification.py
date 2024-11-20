import xgboost as xgb
  

def xgb_classify(X_train, X_test, y_train):

    xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Step 5: Predict on Test Set
    y_pred = xgb_model.predict(X_test)

    return y_pred




