import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, RobustScaler

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]

print("Loading Train and Test datasets...")
train_df = pd.read_csv('KDDTrain+.txt', header=None, names=columns)
test_df = pd.read_csv('KDDTest+.txt', header=None, names=columns)

train_df['target'] = train_df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['target'] = test_df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)

categorical_cols = ['protocol_type', 'service', 'flag']
saved_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    
    saved_encoders[col] = le
    
    test_mapping = {category: idx for idx, category in enumerate(le.classes_)}
    test_df[col] = test_df[col].map(test_mapping).fillna(-1).astype(int)

drop_cols = ['attack_type', 'difficulty_level', 'target']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['target']

print("Scaling numerical features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training the optimized Random Forest model...")
model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=25, 
    random_state=42, 
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

print("Evaluating on Test Dataset...")
y_test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n" + "="*20 + " NEW FINAL TEST RESULTS " + "="*20)
print(f"Optimized Test Dataset Accuracy: {test_accuracy * 100:.2f}%")
print("\nFinal Performance Report:")
print(classification_report(y_test, y_test_pred, target_names=['Normal', 'Attack']))

print("\n" + "="*20 + " EXPORTING PRODUCTION ASSETS " + "="*20)
joblib.dump(model, 'ids_random_forest_model.pkl')
joblib.dump(scaler, 'ids_robust_scaler.pkl')
joblib.dump(saved_encoders, 'label_encoders.pkl')
print("Assets successfully saved: \n -> ids_random_forest_model.pkl \n -> ids_robust_scaler.pkl \n -> label_encoders.pkl")
