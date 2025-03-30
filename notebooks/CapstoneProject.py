#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


# pip install kaggle 
# pip install kagglehub
#run this above command if the kaggle is not installed on your machine


# In[3]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")

print("Path to dataset files:", path)


# In[4]:


df=pd.read_csv(path+"/train.csv")
df_test=pd.read_csv(path+"/test.csv")


# In[5]:


df.head()


# In[6]:


df_test.head()


# In[7]:


df.shape


# In[8]:


df_test.shape


# In[9]:


df.info()


# In[10]:


missing_cols=df.columns[df.isnull().sum()>0]
# Print columns with missing values
print("Columns with Missing Values:\n", missing_cols)


# In[11]:


missing_cols=df.columns[df.isna().sum()>0]
# Print columns with missing values
print("Columns with Missing Values:\n", missing_cols)


# In[12]:


df.describe()


# In[13]:


len(df[df.duplicated()])


# In[14]:


print(df['Activity'].unique())


# In[15]:


#Check the Target Class Imbalance

# Count occurrences of each unique label
label_counts = df['Activity'].value_counts()

# Plot the pie chart
plt.figure(figsize=(6, 6))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title("Distribution of Activity")
get_ipython().run_line_magic('matplotlib', 'inline')
plt.show()


# In[16]:


# Count the number of occurrences for each activity
activity_counts = df['Activity'].value_counts()

# Plot the bar chart
plt.figure(figsize=(10, 6))
activity_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Add labels and title
plt.title('Count of Samples per Activity in HAR Dataset')
plt.xlabel('Activity')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()


# In[17]:


X_initial = df.iloc[:,:-1]
y_initial=df.iloc[:,-1]


# In[18]:


X_train = df.iloc[:,:-1]
y_train =df.iloc[:,-1]

X_test=df_test.iloc[:,:-1]
y_test=df_test.iloc[:,-1]


# In[19]:


from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()

# Fit and transform on X_train

X_train_scaled = scaler.fit_transform(X_train)

# Transform X_test using the same scaler
X_test_scaled = scaler.transform(X_test)



# In[20]:


# Save the scaler to ensure consistent data transformation
joblib.dump(scaler, 'scaler.pkl')


# In[21]:


from sklearn.decomposition import PCA
pca = PCA(n_components=0.95) 
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# Check the explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")
print(f"Number of components to retain 95% variance: {pca.n_components_}")



# In[22]:


X_train_pca.shape


# In[23]:


X_test_pca.shape


# In[24]:


# Save PCA and selected columns
joblib.dump(pca, 'pca_transformer.pkl')


# In[25]:


# assigning proper columns to train dataset 

X_train = X_train_pca
y_train = y_train

X_test=X_test_pca
y_test=y_test


# In[26]:


from sklearn.preprocessing import LabelEncoder
# Encode target labels (Convert categorical to numeric)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)


# Save the encoder for later use
joblib.dump(encoder, 'label_encoder.pkl')


# In[27]:


X_train.shape


# In[28]:


y_train.shape


# In[29]:


X_test.shape


# In[30]:


from sklearn import svm
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[31]:


from sklearn.linear_model import LogisticRegression
# Create and train the Logistic Regression model
logisticregressionmodel = LogisticRegression(max_iter=1000)
logisticregressionmodel.fit(X_train, y_train)

print("Logistic Regression Model Trained Successfully!")


# In[32]:


# Make predictions
y_pred = logisticregressionmodel.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy:.2f}")
print("\n Logistic Regression Model Classification Report:")
print(classification_report(y_test, y_pred))


# In[33]:


# Define parameter grid for tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
    'solver': ['lbfgs', 'saga'],     # Solvers for optimization
    'max_iter': [500, 1000, 2000]    # Number of iterations
}

# Create the GridSearchCV object
#grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', verbose=1)

random_search = RandomizedSearchCV(LogisticRegression(), param_distributions=param_grid,
                                   n_iter=20, cv=5, n_jobs=-1, verbose=2)


# Fit grid search
#grid_search.fit(X_train_scaled, y_train)

random_search.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Cross-Validation Score: {random_search.best_score_:.2f}")

# Use the best estimator to make predictions
best_model_lg = random_search.best_estimator_
y_pred_best = best_model_lg.predict(X_test)

# Evaluate the tuned model
print(classification_report(y_test, y_pred_best))
print(f"Tuned Model Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")


# In[34]:


# Save the trained model to a file
joblib.dump(best_model_lg, 'logistic_model.pkl')


# In[35]:


svc_linear = svm.SVC(kernel='linear')
svc_linear.fit(X_train, y_train)
# make predictions on test set
y_pred=svc_linear.predict(X_test)
# compute and print accuracy score
print('SVC Linear: Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:\n\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print('\n\nConfusion matrix\n\n', cm)


# In[36]:


svc_poly = svm.SVC(kernel='poly')
svc_poly.fit(X_train, y_train)
# make predictions on test set
y_pred=svc_poly.predict(X_test)
# compute and print accuracy score
print('SVC poly: Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:\n\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print('\n\nConfusion matrix\n\n', cm)


# In[37]:


svc_rbf = svm.SVC(kernel='rbf',probability=True,random_state=42)
svc_rbf.fit(X_train, y_train)
# make predictions on test set
y_pred=svc_rbf.predict(X_test)
# compute and print accuracy score
print('SVC RBF: Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

print("\nClassification Report:\n\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print('\n\nConfusion matrix\n\n', cm)


# In[38]:


# Save the model to a .pkl file
joblib.dump(svc_rbf, 'svc_rbf_model.pkl')


# In[39]:


svc_rbf.get_params()


# In[40]:


# Define hyperparameters to tune
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1,'scale'],
    'kernel': ['rbf']
}
svc = svm.SVC()
random_search = RandomizedSearchCV(svc, param_distributions=param_grid,
                                   n_iter=20, cv=5, n_jobs=-1, verbose=2)
random_search.fit(X_train, y_train)

# Best parameters and best score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Accuracy: {random_search.best_score_:.4f}")

# Evaluate on test data
best_model_svc = random_search.best_estimator_
y_pred = best_model_svc.predict(X_test)
print(classification_report(y_test, y_pred))


# In[41]:


import xgboost as xgb

# Create DMatrix for XGBoost (efficient training format)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'multi:softmax',  # Multiclass classification
    'num_class': len(set(y_train)),  # Number of classes
    'eval_metric': 'mlogloss',  # Log loss for multiclass classification
    'max_depth': 6,  # Maximum depth of a tree
    'eta': 0.1,  # Learning rate
    'gamma': 0.1,  # Minimum loss reduction required for split
}

# Train the XGBoost model
num_rounds = 100  # Number of boosting rounds
xgbmodel = xgb.train(params, dtrain, num_boost_round=num_rounds)
print("XGBoost Model Trained Successfully!")


# In[42]:


# Make predictions
y_pred = xgbmodel.predict(dtest)

# Evaluate model 
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[43]:


# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0.5, 1, 2]
}

# Initialize XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='logloss')

# Grid Search with 5-fold cross-validation
#grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,  cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

rand_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=20, cv=5, n_jobs=-1, verbose=2)


# Fit GridSearchCV
rand_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {rand_search.best_params_}")

# Use best model from GridSearchCV
best_model_xgboost = rand_search.best_estimator_

# Make predictions
y_pred_best = best_model_xgboost.predict(X_test)

# Evaluate performance
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Tuned Model Accuracy: {accuracy_best:.4f}")


# In[44]:


# Save the model to a .pkl file
joblib.dump(best_model_xgboost, 'xgbmodel.pkl')


# In[45]:


from sklearn.ensemble import RandomForestClassifier
# Create and train the Random Forest model
randomforestmodel = RandomForestClassifier(n_estimators=100, random_state=42)
randomforestmodel.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")


# In[46]:


# Make predictions
y_pred = randomforestmodel.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f" Random Forest Model Accuracy: {accuracy:.2f}")
print("\n Random Forest Model Classification Report:")
print(classification_report(y_test, y_pred))


# In[47]:


randomforestmodel.get_params()


# In[48]:


# Define the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Perform Grid Search with cross-validation
#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,cv=5, n_jobs=-1, verbose=2, scoring='accuracy')


rand_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=20, cv=5, n_jobs=-1, verbose=2)

# Fit the model to training data
#grid_search.fit(X_train, y_train)
rand_search.fit(X_train, y_train)

# Get the best model
#best_rf = grid_search.best_estimator_
#print(f"Best parameters: {grid_search.best_params_}")


# Get the best model
best_rf = rand_search.best_estimator_
print(f"Best parameters: {rand_search.best_params_}")

# Make predictions
y_pred = best_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


# In[49]:


# Save the model to a .pkl file
joblib.dump(randomforestmodel, 'randomforestmodel.pkl')


# In[50]:


import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.optimizers import Adam


# In[51]:


# Define model
def create_model(neurons=64, activation='relu', dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(X_train.shape[1],), activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # Multi-class classification
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



annmodel = KerasClassifier(model=create_model,neurons=64, activation='relu', dropout_rate=0.2, learning_rate=0.001, verbose=0)





# In[52]:


# Train the model
annmodel.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)
annmodel.model_.summary()


# In[53]:


# Evaluate on test data
test_loss, test_acc = annmodel.model_.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


# In[54]:


annmodel.get_params().keys()


# In[55]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Randomized search
param_dist = {
    'neurons': [32, 64, 128],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.2, 0.3, 0.5],
    'batch_size': [32, 64],
    'epochs': [50, 100],
    'learning_rate': uniform(0.0001, 0.01)
}

random_search = RandomizedSearchCV(estimator=annmodel, param_distributions=param_dist,n_iter=20, cv=3, n_jobs=-1, verbose=2)
 
random_search.fit(X_train, y_train)


# Get the best model
best_annmodel = random_search.best_estimator_
print(f"Best parameters: {random_search.best_params_}")

# Make predictions
y_pred = best_annmodel.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))



# In[56]:


#!pip install scikeras


# In[57]:


# Save the model to a .pkl file
joblib.dump(best_annmodel, 'ann_model.pkl')


# In[58]:


#Ensemble of models
from sklearn.ensemble import  VotingClassifier

# Create an ensemble using VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', randomforestmodel),
        ('svc', svc_rbf),
        ('xgb', best_model_xgboost),
        ('lg',best_model_lg),
        ('ann',best_annmodel)
    ],
    voting='soft'  # Use 'soft' for probability-based voting or 'hard' for majority voting
)

# Fit the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions
y_pred = ensemble_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[59]:


joblib.dump(ensemble_model, 'ensemble_model.pkl')


# In[61]:


best_annmodel.model_.summary()


# In[62]:


best_annmodel.model_.get_config()


# In[63]:


# Get model weights
weights = best_annmodel.model_.get_weights()

# Print summary of weights and biases
for layer_index, layer in enumerate(best_annmodel.model_.layers):
    print(f"Layer {layer_index + 1}: {layer.name}")
    layer_weights = layer.get_weights()
    
    if len(layer_weights) > 0:
        weights, biases = layer_weights
        print(f"  - Weights Shape: {weights.shape}")
        print(f"  - Biases Shape: {biases.shape}")
    else:
        print("  - No trainable weights in this layer.")


# In[ ]:




