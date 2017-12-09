#
# classifier.py
# 
# Wrapper functions that employ sklearn's support vector classifier and cross validation tools
# to train and evaluate a support vector classifier on the given data. 
#

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt


def EvaluateClassifier(X, Y, C, weight):
	'''
	Input: 
		* Dataset X
		* Labels Y
	Output:
		* outputs performance metrics: accuracy, sens, spec
	
	Evaluates a linear-kernel support classifer with given penalty input C and melanoma class-weight weight
	with 10-fold cross-validation. We must adjust the class weight to account for the class imbalance inherent 
	to the dataset. 
	'''

	svm_man = SVC(C = C, kernel='linear', class_weight={1:weight})

	scores = cross_val_score(svm_man, X, Y, cv=10, scoring='accuracy')
	acc = scores.mean()
	print 'acc is:', acc

	#------ Compute metrics ---------
	y_predict = cross_val_predict(svm_man, X, Y, cv=10)
	conf_mat = confusion_matrix(Y, y_predict)
	tn, fp, fn, tp = conf_mat.ravel()
	sens = float(tp)/(tp+fn)
	spec = float(tn)/(tn+fp)
	print 'sens is:', sens
	print 'spec is:', spec
	print conf_mat

	return (acc, sens, spec)


def VisualizeTuning(grid, X_bounds, Y_bounds):

    (X_low, X_high, X_step) = X_bounds
    Y_low, Y_high, Y_step = Y_bounds

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(grid, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('Melanoma Class Weight')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(8), np.arange(X_low, X_high, X_step))
    plt.yticks(np.arange(8), np.arange(Y_low, Y_high, Y_step))
    plt.title('10-fold CV Accuracy')
    plt.show()

