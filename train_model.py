import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

#PATHS
DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#DATA
train_path = os.path.join(DATA_DIR, 'train.csv')
test_path = os.path.join(DATA_DIR, 'test.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

#FEATURES
all_features = [
    'Geography', 'Gender', 'CreditScore', 'Age', 'Tenure',
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

X_train = train_df[all_features].copy()
y_train = train_df['Exited']
X_test = test_df[all_features].copy()

#ENCODE
le_geo = LabelEncoder()
le_gen = LabelEncoder()

X_train['Geography'] = le_geo.fit_transform(X_train['Geography'])
X_train['Gender'] = le_gen.fit_transform(X_train['Gender'])

X_test['Geography'] = le_geo.transform(X_test['Geography'])
X_test['Gender'] = le_gen.transform(X_test['Gender'])

#SCALE
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

#TRAIN
dt = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced'
)
dt.fit(X_train, y_train)

#SAVE
joblib.dump(dt, os.path.join(MODEL_DIR, 'model_dt.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(le_geo, os.path.join(MODEL_DIR, 'le_geo.pkl'))
joblib.dump(le_gen, os.path.join(MODEL_DIR, 'le_gen.pkl'))
joblib.dump(all_features, os.path.join(MODEL_DIR, 'feature_order.pkl'))  # ← CRITICAL

print("All models & feature order saved to /models/")

#PREDICT&SAVE(FOR KAGGLE)
y_pred_proba = dt.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'Exited': y_pred_proba
})
submission_path = os.path.join(DATA_DIR, 'submission.csv')
submission.to_csv(submission_path, index=False)
print(f"submission.csv saved to {submission_path}")

#TRAIN METRICS
y_train_pred = dt.predict(X_train)
print(f"\nTrain Accuracy: {accuracy_score(y_train, y_train_pred):.3f}")
print(classification_report(y_train, y_train_pred))