import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error,mean_squared_error,
                             r2_score,confusion_matrix,accuracy_score)

df = pd.read_csv(r"D:\5th sem'\pythonproject5\Air_Quality.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nNull Values Before Cleaning:\n", df.isnull().sum())

# OBJECTIVE 1: DATA CLEANING & PREPROCESSING

df['Message'] = df['Message'].fillna("No Message")
df['Data Value'] = df['Data Value'].fillna(df['Data Value'].median())
df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
df['Year'] = df['Start_Date'].dt.year
df['Month'] = df['Start_Date'].dt.month
def get_season(x):
    x = str(x).lower()
    if 'winter' in x: return 'Winter'
    if 'summer' in x: return 'Summer'
    if 'spring' in x: return 'Spring'
    if 'fall' in x: return 'Fall'
    return 'Annual'

df['Season'] = df['Time Period'].apply(get_season)
df = df.dropna(subset=['Year', 'Month', 'Data Value'])

print("\nNull Values After Cleaning:\n", df.isnull().sum())
print("Final Dataset Shape:", df.shape)

# OBJECTIVE 2: EXPLORATORY DATA ANALYSIS (EDA)

print("\nEDA Summary of Pollution Values:")
print(df['Data Value'].describe())

plt.figure(figsize=(7,4))
sns.histplot(df['Data Value'], kde=True)
plt.title("EDA: Pollution Value Distribution")
plt.xlabel("Pollution Value")
plt.ylabel("Frequency")
plt.show()
X = df[['Name', 'Geo Type Name', 'Geo Place Name',
        'Season', 'Year', 'Month']]
y = df['Data Value']
ct = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'),
     ['Name', 'Geo Type Name', 'Geo Place Name', 'Season'])], remainder='passthrough')

# OBJECTIVE 3: REGRESSION (RANDOM FOREST)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
reg_model = Pipeline([
    ('preprocess', ct),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

print("\nRegression Results:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2  :", r2_score(y_test, y_pred))

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Pollution Value")
plt.ylabel("Predicted Pollution Value")
plt.title("Regression: Actual vs Predicted")
plt.show()

# OBJECTIVE 4: CLASSIFICATION (KNN ONLY)

df['Pollution_Level'] = pd.qcut(
    df['Data Value'], 3, labels=['Low', 'Medium', 'High']
)
y_clf = df['Pollution_Level']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

knn_model = Pipeline([
    ('preprocess', ct),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_train_c, y_train_c)
y_pred_knn = knn_model.predict(X_test_c)

acc_knn = accuracy_score(y_test_c, y_pred_knn)
cm_knn = confusion_matrix(
    y_test_c, y_pred_knn,
    labels=['Low', 'Medium', 'High']
)

print("\nKNN Classification Results:")
print("Accuracy:", acc_knn)
print("Confusion Matrix:\n", cm_knn)

plt.figure(figsize=(6,5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# DECISION TREE (ONLY VISUALIZATION – NO CLASSIFICATION)

dt_visual = Pipeline([
    ('preprocess', ct),
    ('classifier', DecisionTreeClassifier(
        max_depth=4, random_state=42))
])

dt_visual.fit(X, y_clf)

dt_clf = dt_visual.named_steps['classifier']
feature_names = dt_visual.named_steps[
    'preprocess'].get_feature_names_out()

plt.figure(figsize=(22,10))
plot_tree(
    dt_clf,
    feature_names=feature_names,
    class_names=['Low', 'Medium', 'High'],
    filled=True
)
plt.title("Decision Tree Visualization (Structure Only)")
plt.show()

#objective 5: corelation

corr = df[['Data Value','Year','Month']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# OBJECTIVE 6: CLUSTERING (K-MEANS)

cluster_data = df[['Data Value', 'Year']]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(cluster_data)

print("\nClustering Results:")
print("Cluster Centers:\n", kmeans.cluster_centers_)

plt.figure(figsize=(7,5))
plt.scatter(cluster_data['Data Value'],
            cluster_data['Year'],
            c=df['Cluster'])
plt.xlabel("Pollution Value")
plt.ylabel("Year")
plt.title("K-Means Clustering of Pollution Data")
plt.show()

















