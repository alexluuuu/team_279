#
# classifier.py
# 
# Wrapper function that employs sklearn's support vector classifier and cross validation tools
# to train and evaluate a support vector classifier on the given data. 
#

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import matplotlib.pyplot as plt

def Tuner(combined_features, groundtruth):
    '''
    Input: 
        * all features
        * ground truth labels

    Output:
        * tuning grid of (accuracy, sensitivity, specificity) for specified bounds on C, class_weight

    '''
    C = .5
    C_bound = 8
    W_bound = 8
    grid = np.zeros((C_bound, W_bound, 3))
    for i in range(C_bound):
        weight = 1.5
        for j in range(W_bound):
            print 'for C', C
            print 'for weight', weight
            grid[i,j] = EvaluateClassifier(combined_features, groundtruth, C, weight)
            weight += .25
            print '---------------------------'
        C += .5
    return grid

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


