#DO A CLASSIFY

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import numpy as np


def RunClassifier(X, Y):

	svm_man = SVC()
	scores = cross_val_score(svm_man, X, Y, cv=LeaveOneOut(), scoring='accuracy')
	print scores.mean()
	y_predict = cross_val_predict(svm_man, X, Y, cv=LeaveOneOut())
	print confusion_matrix(Y, y_predict)