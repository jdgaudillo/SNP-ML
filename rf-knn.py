import warnings
warnings.filterwarnings('ignore')
import glob
import numpy as np
import pandas as pd
import itertools


from datetime import datetime
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

"""
	This script implements rf-knn on the SNP dataset.

	Input:
		file: str
			Encoded dataset
	
"""


start = datetime.now()
data_file = 'dataset/encoded_snp.csv'
data = pd.read_csv(data_file, sep = ' ')
data = data.drop('usersid', axis=1)
data = data.replace(np.NaN, 0)

correct = 0
mismatch = 0
y_pred = []
n_features = 400
k = 7
counter = 0
n_samples = 128
y_true = []


snp = data.columns.values
snp = snp[:-1]
X = data[snp].values

def convertLabel(data):
	label_maker = LabelEncoder()
	data['Encode'] = label_maker.fit_transform(data['Class'])

def knn(X_train, y_train, X_test):
	scaler = StandardScaler()  
	scaler.fit(X_train)

	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test)

	classifier = KNeighborsClassifier(n_neighbors=k)  
	classifier.fit(X_train, y_train) 
	return classifier.predict(X_test)

convertLabel(data)
y = data['Encode'].values

loo = LeaveOneOut()
loo.get_n_splits(X)


for train_index, test_index in loo.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	fs = RandomForestClassifier(max_features = 500, n_estimators=200, random_state=42)
	fs.fit(X_train, y_train)
	ranking = dict(zip(snp, fs.feature_importances_))
	ranking = dict(itertools.islice(ranking.items(), n_features))
	top_snp = list(ranking.keys())
	top_snp = np.asarray(top_snp)

	X_train = data.drop(test_index)
	X_train = X_train[top_snp].values
	X_test = data.reindex(test_index)
	X_test = X_test[top_snp].values

	tmp = knn(X_train, y_train, X_test)
	
	y_pred.append(tmp[0])
	y_true.append(y_test)
	print (counter, '===========================')
	print('Prediction:', tmp)
	print('True label', y_test[0])

	counter = counter + 1
	

y_pred = np.asarray(y_pred)
y_true = np.asarray(y_true)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print('Number of SNPs:', n_features)
print('Accuracy using of rf-knn', accuracy)
print('Precision using of rf-knn', accuracy)
print('Recall using of rf-knn', accuracy)
print('\nConfusion matrix of k-nearest neighbors (kNN) optimized on the test data:')
print(pd.DataFrame(confusion_matrix(y_true, y_pred),
      columns=['pred_pos', 'pred_neg'], index=['pos', 'neg']))
print('Run-time', datetime.now() - start)
    

