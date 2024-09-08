import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Changeable parameters
target = "diagnosis"
test_size = 0.2

# Load and preprocess the data
df = pd.read_csv("HEART_MODEL/Arrhythmia/arrhythmia_dataset.csv")
df.dropna(axis=0, inplace=True)
df.drop(df.columns[20:-2], axis=1, inplace=True)

# Convert diagnosis to binary classification
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x > 1 else 0)

# Determine if the task is classification or regression
if isinstance(df[target][0], (str, int, np.int64)):
    models_type = 'classification'
else:
    models_type = 'regression'

# Label encoding for categorical variables
def label_encoding(old_column):
    le = LabelEncoder()
    le.fit(old_column)
    new_column = le.transform(old_column)
    return new_column

for i in df.columns:
    if isinstance(df[i][0], str):
        df[i] = label_encoding(df[i])

# Extracting X and y
y = df[target].values
x = df.drop([target], axis=1).values

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=test_size, random_state=42)

# Ensemble models for classification
if models_type == 'classification':
    # Define the models
    model1 = RandomForestClassifier(n_estimators=100, random_state=42)
    model2 = XGBClassifier(eval_metric='mlogloss', random_state=42)
    model3 = LogisticRegression(max_iter=1000, random_state=42)
    model4 = KNeighborsClassifier()
    model5 = SVC(probability=True, random_state=42)

    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', model1),
            ('xgb', model2),
            ('lr', model3),
            ('knn', model4),
            ('svc', model5)
        ],
        voting='soft',
        weights=[2, 3, 1, 1, 2]
    )

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('HEART_MODEL/Arrhythmia/arrhythmia_prediction.pkl', 'wb') as model_file:
    pickle.dump(ensemble_model, model_file)

print("Model trained and saved successfully.")
