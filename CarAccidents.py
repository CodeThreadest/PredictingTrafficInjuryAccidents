import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Traffic_Crashes(1).csv')

# Inspect initial dataset
print("Initial Dataset Info:")
print(df.info())

# Encode target variable as binary accident indicator
df['accident'] = df['INJURIES_TOTAL'].apply(lambda x: 1 if x > 2 else 0)

# Drop irrelevant or high-missing columns
columns_to_drop = [
    'INJURIES_TOTAL', 'CRASH_DATE_EST_I', 'LANE_CNT', 'NOT_RIGHT_OF_WAY_I', 
    'PHOTOS_TAKEN_I', 'STATEMENTS_TAKEN_I', 'DOORING_I', 'WORK_ZONE_I', 
    'WORK_ZONE_TYPE', 'WORKERS_PRESENT_I', 'INJURIES_FATAL', 'INJURIES_INCAPACITATING',
    'INJURIES_NON_INCAPACITATING', 'INJURIES_REPORTED_NOT_EVIDENT',
    'INJURIES_NO_INDICATION', 'INJURIES_UNKNOWN', 'MOST_SEVERE_INJURY', 'CRASH_TYPE'
]
df.drop(columns=columns_to_drop, inplace=True)

# Convert 'CRASH_DATE' to datetime and extract date components
df['CRASH_DATE'] = pd.to_datetime(df['CRASH_DATE'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['CRASH_YEAR'] = df['CRASH_DATE'].dt.year
df['CRASH_MONTH'] = df['CRASH_DATE'].dt.month
df['CRASH_DAY'] = df['CRASH_DATE'].dt.day
df['CRASH_HOUR'] = df['CRASH_DATE'].dt.hour

# Handle missing values with forward fill
df.ffill(inplace=True)

# Define categorical, binary, and numerical features
categorical_features = [
    'REPORT_TYPE', 'FIRST_CRASH_TYPE', 'DAMAGE', 'PRIM_CONTRIBUTORY_CAUSE', 
    'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'WEATHER_CONDITION', 
    'LIGHTING_CONDITION', 'TRAFFICWAY_TYPE', 'ALIGNMENT', 'ROADWAY_SURFACE_COND', 
    'ROAD_DEFECT', 'STREET_DIRECTION'
]
binary_features = ['HIT_AND_RUN_I', 'INTERSECTION_RELATED_I']
numerical_features = ['POSTED_SPEED_LIMIT', 'NUM_UNITS', 'CRASH_HOUR', 'CRASH_DAY', 'CRASH_MONTH']

# Standardize binary features to 0 and 1
for col in binary_features:
    df[col] = df[col].apply(lambda x: 1 if x == 'Y' else 0)

# Impute missing values for numerical and categorical features
num_imputer = SimpleImputer(strategy='mean')
df[numerical_features] = num_imputer.fit_transform(df[numerical_features])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])

# OneHotEncode categorical features
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = one_hot_encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(categorical_features))

# Combine encoded features with the dataset and drop original categorical columns
df_encoded = pd.concat([df, encoded_df], axis=1)
df_encoded.drop(columns=categorical_features, inplace=True)

# Drop non-numeric or irrelevant columns explicitly
non_numeric_columns_to_drop = ['DATE_POLICE_NOTIFIED', 'SEC_CONTRIBUTORY_CAUSE', 'STREET_NAME']
df_encoded.drop(columns=non_numeric_columns_to_drop, inplace=True)

# Normalize numerical features
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# Define features (X) and target (y)
y = df_encoded['accident']
X = df_encoded.drop(['accident', 'CRASH_RECORD_ID', 'LOCATION', 'CRASH_DATE'], axis=1)

# Check for non-numeric columns in X
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
if not non_numeric_cols.empty:
    print(f"Warning: Non-numeric columns found in X: {non_numeric_cols}")
else:
    print("All columns in X are numeric.")

# Ensure no missing values in the feature matrix
X.fillna(X.mean(), inplace=True)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train a decision tree classifier
decision_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
decision_tree.fit(X_train, y_train)

# Predict on the test set
y_pred = decision_tree.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)

# Plot feature importance
feature_importance = decision_tree.feature_importances_
important_features = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Features by Importance:")
print(important_features.head(10))

plt.figure(figsize=(20, 15))
plt.barh(important_features['Feature'], important_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Top Features by Importance')
plt.show()

# Plot decision tree
plt.figure(figsize=(30, 10))
plot_tree(decision_tree, filled=True, feature_names=X.columns, class_names=['No Accident', 'Accident'], fontsize=10)
plt.show()
