#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

# Load training dataset
df = pd.read_csv('train.csv')

# Check class distribution
df.sample(5, random_state=10)
df['Personality'].value_counts()

# Imputation modules
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

# Dataset information
df.info()

# Convert boolean strings to integers
def boolToint(a):
    if (pd.isna(a) == False):    
        if(a.lower() == 'no'):
            return 0
        else:
            return 1

df['Stage_fear'] = df['Stage_fear'].apply(boolToint)
df['Drained_after_socializing'] = df['Drained_after_socializing'].apply(boolToint)

# Dataset information after conversion
df.info()
df.describe()

# Check for duplicate rows
df.duplicated().value_counts()

# Drop ID column
df = df.drop(columns=['id'])
df.sample(5, random_state = 10)

# Normalize numerical features
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
df.iloc[:, 0:7] = scaler.fit_transform(df.iloc[: , 0:7])
df.sample(5, random_state= 10)

# Split features and target
X= df.iloc[:, 0:7]
y= df.iloc[:, 7]

# Handle missing values using KNN imputer
imp = KNNImputer(n_neighbors=5)
imp.set_output(transform = "pandas")
X = imp.fit_transform(X)
X.sample(5, random_state=10)

# Encode target labels
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y = lb.fit_transform(y)
lb.classes_
np.unique(y)

# Train-test split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=4)
X_train.shape

# Train Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=14,
    criterion='entropy',
    min_samples_split=10
)
rf.fit(X_train, y_train)

# Evaluate on test set
from sklearn.metrics import classification_report
pred = rf.predict(X_test)
print(classification_report(y_test, pred))

# Evaluate on training set
pred2 = rf.predict(X_train)
print(classification_report(y_train, pred2))

# Load test dataset
subm_df = pd.read_csv('test.csv')
subm_df['Stage_fear'] = subm_df['Stage_fear'].apply(boolToint)
subm_df['Drained_after_socializing'] = subm_df['Drained_after_socializing'].apply(boolToint)

scaler = MinMaxScaler()
subm_df.iloc[:, 1:] = scaler.fit_transform(subm_df.iloc[: , 1:])
subm_df.info()

subm_df2 = imp.fit_transform(subm_df.iloc[:,1:])
pred_subm = rf.predict(subm_df2)

# Convert predictions back to original labels
pred_subm2 = lb.classes_[pred_subm]

# Save predictions to CSV
subm_final_df = pd.DataFrame({
    'id': subm_df['id'],
    'Personality': pred_subm2
})
subm_final_df = subm_final_df.set_index('id')
subm_final_df.to_csv('final_submission6.csv')