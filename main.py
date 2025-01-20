import pandas as pd
import matplotlib.pyplot as plt

#Initalize the data
df = pd.read_csv('creditcard.csv')

# Visualize the data
df['Class'].value_counts()
df.hist(bins=30, figsize=(30,30))
df.describe()

# Preprocess the data
from sklearn.preprocessing import RobustScaler
new_df = df.copy()
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].to_numpy.reshape(-1,1))
time = new_df['Time']
new_df['Time'] = (time - time.min()) / (time.max() - time.min()) 
new_df = new_df.sample(frac=1, random_state=1)
train, test, val = new_df[:240000], new_df[240000:262000], new_df[262000:]
train['Class'].value_counts(), test['Class'].value_counts(), val['Class'].value_counts()   
train_np, test_np, val_np = train.to_numpy(), test.to_numpy(), val.to_numpy()
train_np.shape, test_np.shape, val_np.shape

x_train, y_train = train_np[:,:-1], train_np[:,-1]
x_test, y_test = test_np[:,:-1], test_np[:,-1]
x_val, y_val = val_np[:,:-1], val_np[:,-1]
x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape

# Build the models

# Regression model
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)
logistic_model.score(x_test, y_test)

from sklearn.metrics import classification_report
print(classification_report(y_val, logistic_model.predict(x_val), target_names=['Not Fraud', 'Fraud']))

# Shallow Neural Network model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

shallow_nn = Sequential()
shallow_nn.add(InputLayer((x_train.shape[1],)))
shallow_nn.add(Dense(2, 'relu'))
shallow_nn.add(BatchNormalization()) # not necessary but helps learn the data better
shallow_nn.add(Dense(1, 'sigmoid'))

checkpoint = ModelCheckpoint('shallow_nn', save_best_only=True)
shallow_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
shallow_nn.summary()
shallow_nn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, callbacks=checkpoint)

def neural_net_predictions(model, x):
  return (model.predict(x).flatten() > 0.5).astype(int)
neural_net_predictions(shallow_nn, x_val)
print(classification_report(y_val, neural_net_predictions(shallow_nn, x_val), target_names=['Not Fraud', 'Fraud']))

# Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=2, n_jobs=-1)
rf.fit(x_train, y_train)
print(classification_report(y_val, rf.predict(x_val), target_names=['Not Fraud', 'Fraud']))

# Gradient Boosting model
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
gbc.fit(x_train, y_train)
print(classification_report(y_val, gbc.predict(x_val), target_names=['Not Fraud', 'Fraud']))

# Linear support vector machine model
from sklearn.svm import LinearSVC
svc = LinearSVC(class_weight='balanced') # class_weight='balanced' helps with imbalanced data
svc.fit(x_train, y_train)
print(classification_report(y_val, svc.predict(x_val), target_names=['Not Fraud', 'Fraud']))

# Balance the data to improve the models
not_frauds = new_df.query('Class == 0')
frauds = new_df.query('Class == 1')
not_frauds['Class'].value_counts(), frauds['Class'].value_counts()
balanced_df = pd.concat([frauds, not_frauds.sample(len(frauds), random_state=1)])
balanced_df = balanced_df.sample(frac=1, random_state=1)

balanced_df_np = balanced_df.to_numpy()

x_train_b, y_train_b = balanced_df_np[:700, :-1], balanced_df_np[:700, -1].astype(int)
x_test_b, y_test_b = balanced_df_np[700:842, :-1], balanced_df_np[700:842, -1].astype(int)
x_val_b, y_val_b = balanced_df_np[842:, :-1], balanced_df_np[842:, -1].astype(int)
x_train_b.shape, y_train_b.shape, x_test_b.shape, y_test_b.shape, x_val_b.shape, y_val_b.shape
pd.Series(y_train_b).value_counts(), pd.Series(y_test_b).value_counts(), pd.Series(y_val_b).value_counts()

# New models with balanced data labeeled with _b

# logistic regression model
logistic_model_b = LogisticRegression()
logistic_model_b.fit(x_train_b, y_train_b)
print(classification_report(y_val_b, logistic_model_b.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))

# shallow neural network model (Likely to overfit for training model so I switched to 1 relu layer)
shallow_nn_b = Sequential()
shallow_nn_b.add(InputLayer((x_train_b.shape[1],)))
shallow_nn_b.add(Dense(1, 'relu'))
shallow_nn_b.add(BatchNormalization())
shallow_nn_b.add(Dense(1, 'sigmoid'))

checkpoint = ModelCheckpoint('shallow_nn_b', save_best_only=True)
shallow_nn_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
shallow_nn_b.fit(x_train_b, y_train_b, validation_data=(x_val_b, y_val_b), epochs=40, callbacks=checkpoint)
shallow_nn_b.fit(x_train_b, y_train_b, validation_data=(x_val_b, y_val_b), epochs=40, callbacks=checkpoint)
print(classification_report(y_val_b, neural_net_predictions(shallow_nn_b, x_val_b), target_names=['Not Fraud', 'Fraud']))

# random forest model
rf_b = RandomForestClassifier(max_depth=2, n_jobs=-1)
rf_b.fit(x_train_b, y_train_b)
print(classification_report(y_val_b, rf.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))

# gradient boosting model
gbc_b = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0)
gbc_b.fit(x_train_b, y_train_b)
print(classification_report(y_val_b, gbc.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))

# linear support vector machine model
svc_b = LinearSVC(class_weight='balanced')
svc_b.fit(x_train_b, y_train_b)
print(classification_report(y_val_b, svc.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))


# Visualize the data
plt.figure(figsize=(10,10))
plt.scatter(x_val_b[:,0], x_val_b[:,1], c=y_val_b, cmap='coolwarm', alpha=0.5)
plt.title('Actual Data')
plt.show()

# Visualize the predictions
plt.figure(figsize=(10,10))
plt.scatter(x_val_b[:,0], x_val_b[:,1], c=neural_net_predictions(shallow_nn_b, x_val_b), cmap='coolwarm', alpha=0.5)
plt.title('Predicted Data')
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(x_val_b[:,0], x_val_b[:,1], c=rf_b.predict(x_val_b), cmap='coolwarm', alpha=0.5)
plt.title('Predicted Data')
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(x_val_b[:,0], x_val_b[:,1], c=gbc_b.predict(x_val_b), cmap='coolwarm', alpha=0.5)
plt.title('Predicted Data')
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(x_val_b[:,0], x_val_b[:,1], c=svc_b.predict(x_val_b), cmap='coolwarm', alpha=0.5)
plt.title('Predicted Data')
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(x_val_b[:,0], x_val_b[:,1], c=logistic_model_b.predict(x_val_b), cmap='coolwarm', alpha=0.5)
plt.title('Predicted Data')
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(x_val_b[:,0], x_val_b[:,1], c=y_val_b, cmap='coolwarm', alpha=0.5)
plt.title('Actual Data')
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(x_val_b[:,0], x_val_b[:,1], c=neural_net_predictions(shallow_nn_b, x_val_b), cmap='coolwarm', alpha=0.5)
plt.title('Predicted Data')
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(x_val_b[:,0], x_val_b[:,1], c=rf_b.predict(x_val_b), cmap='coolwarm', alpha=0.5)
plt.title('Predicted Data')
plt.show()

# Compare the models
from sklearn.metrics import roc_auc_score
roc_auc_score(y_val_b, logistic_model_b.predict(x_val_b)), roc_auc_score(y_val_b, neural_net_predictions(shallow_nn_b, x_val_b)), roc_auc_score(y_val_b, rf_b.predict(x_val_b)), roc_auc_score(y_val_b, gbc_b.predict(x_val_b)), roc_auc_score(y_val_b, svc_b.predict(x_val_b))

# Conclusion
# The shallow neural network model seems to be the best model for this dataset. 
# It has the highest AUC score and the best precision and recall for fraud detection. 
# The random forest model is the second best model for this dataset. 
# The logistic regression model is the worst model for this dataset. 
# The gradient boosting model and the linear support vector machine model are in the middle.