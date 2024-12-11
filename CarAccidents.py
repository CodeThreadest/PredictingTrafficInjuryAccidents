import os
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

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Encode target variable as binary accident indicator
df['accident'] = df['INJURIES_TOTAL'].apply(lambda x: 1 if x > 2 else 0)

# Drop irrelevant or high-missing columns
columns_to_drop = [
     'REPORT_TYPE', 'DEVICE_CONDITION', 'NUM_UNITS', 'NUM_UNITS','TRAFFIC_CONTROL_DEVICE',
    'INJURIES_TOTAL', 'CRASH_DATE_EST_I', 'LANE_CNT', 'NOT_RIGHT_OF_WAY_I', 'CRASH_DATE','DAMAGE',
    'PHOTOS_TAKEN_I', 'STATEMENTS_TAKEN_I', 'DOORING_I', 'WORK_ZONE_I', 'STREET_NO', 'LATITUDE',
    'WORK_ZONE_TYPE', 'WORKERS_PRESENT_I', 'INJURIES_FATAL', 'INJURIES_INCAPACITATING', 'CRASH_MONTH',
    'INJURIES_NON_INCAPACITATING', 'INJURIES_REPORTED_NOT_EVIDENT', 'BEAT_OF_OCCURRENCE',
    'INJURIES_NO_INDICATION', 'INJURIES_UNKNOWN', 'MOST_SEVERE_INJURY', 'CRASH_TYPE', 'CRASH_HOUR', 'CRASH_DAY_OF_WEEK'
]
df.drop(columns=columns_to_drop, inplace=True)

# Handle missing values with forward fill
df.ffill(inplace=True)

# Define categorical, binary, and numerical features
categorical_features = [
    'FIRST_CRASH_TYPE', 'PRIM_CONTRIBUTORY_CAUSE', 
     'WEATHER_CONDITION', 
    'LIGHTING_CONDITION', 'TRAFFICWAY_TYPE', 'ALIGNMENT', 'ROADWAY_SURFACE_COND', 
    'ROAD_DEFECT', 'STREET_DIRECTION'
]
binary_features = ['HIT_AND_RUN_I', 'INTERSECTION_RELATED_I']
numerical_features = ['POSTED_SPEED_LIMIT']

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
X = df_encoded.drop(['accident', 'CRASH_RECORD_ID', 'LOCATION'], axis=1)

# Shorten feature names for visualization
short_feature_names = {name: f'F{i}' for i, name in enumerate(X.columns)}
X_short = X.rename(columns=short_feature_names)

# Check for non-numeric columns in X
non_numeric_cols = X_short.select_dtypes(exclude=[np.number]).columns
if not non_numeric_cols.empty:
    print(f"Warning: Non-numeric columns found in X: {non_numeric_cols}")
else:
    print("All columns in X are numeric.")

# Ensure no missing values in the feature matrix
X_short.fillna(X_short.mean(), inplace=True)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_short, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train the pruned decision tree with ccp_alpha
ccp_alpha_fixed = 2e-06
pruned_tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha_fixed)
pruned_tree.fit(X_train, y_train)

# Evaluate the pruned tree
y_pred_pruned = pruned_tree.predict(X_test)
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
report_pruned = classification_report(y_test, y_pred_pruned)
conf_matrix_pruned = confusion_matrix(y_test, y_pred_pruned)

print("\nPruned Model Evaluation Metrics:")
print(f"Accuracy: {accuracy_pruned}")
print("Classification Report:")
print(report_pruned)
print("Confusion Matrix:")
print(conf_matrix_pruned)

# Extract feature importances with original feature names
feature_importances = pruned_tree.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,  # Use original feature names here
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Get the top 20 features (original names)
top_20_features = importance_df.head(20)

# Print the top 20 important features
print("\nTop 20 Important Features:")
print(top_20_features[['Feature', 'Importance']])

# Use shortened names only for tree visualization
short_feature_names = {name: f'F{i}' for i, name in enumerate(X.columns)}
short_feature_name_list = [short_feature_names[name] for name in X.columns]

# Plot the top 20 important features using a horizontal bar plot
plt.figure(figsize=(12, 8))
plt.barh(top_20_features['Feature'], top_20_features['Importance'], align='center')
plt.xlabel('Feature Importance', fontsize=14)
plt.ylabel('Feature Name', fontsize=14)
plt.title('Top 20 Important Features', fontsize=16)
plt.gca().invert_yaxis()  # Invert y-axis to show the highest importance at the top
plt.tight_layout()
plt.show()
# Plot the pruned decision tree using shortened names
plt.figure(figsize=(40, 20))
plot_tree(
    pruned_tree,
    feature_names=short_feature_name_list,  # Use shortened names here
    class_names=['No Accident', 'Accident'],
    filled=True,
    proportion=True,
    rounded=True,
    precision=2
)
plt.title('Pruned Decision Tree (Shortened Names)')
plt.show()
