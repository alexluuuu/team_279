#DO A CLASSIFY

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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

def TuneClassifier(X, Y):
	
	C_range = np.logspace(-1, 6, 8)
	#weight_range = np.logspace(1, 10, 10)
	param_grid = dict(C=C_range)
	cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
	grid = GridSearchCV(SVC(kernel='linear', class_weight="balanced"), param_grid=param_grid, cv=cv)
	grid.fit(X, Y)

	print("The best parameters are %s with a score of %0.2f"
	      % (grid.best_params_, grid.best_score_))
	
	# Now we need to fit a classifier for all parameters in the 2d version
	# (we use a smaller set of parameters here because it takes a while to train)
	print X.shape

	'''
	C_2d_range = [1e-2, 1, 1e2]
	gamma_2d_range = [1e-1, 1, 1e1]
	classifiers = []
	for C in C_2d_range:
	    for gamma in gamma_2d_range:
	        clf = SVC(C=C, gamma=gamma)
	        clf.fit(X, Y)
	        classifiers.append((C, gamma, clf))
	'''
	# #############################################################################
	# Visualization
	#
	# draw visualization of parameter effects
	'''
	plt.figure(figsize=(8, 6))
	xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
	for (k, (C, gamma, clf)) in enumerate(classifiers):
	    # evaluate decision function in a grid
	    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
	    Z = Z.reshape(xx.shape)

	    # visualize decision function for these parameters
	    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
	    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
	              size='medium')

	    # visualize parameter's effect on decision function
	    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
	    plt.scatter(X[:, 0], X[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
	                edgecolors='k')
	    plt.xticks(())
	    plt.yticks(())
	    plt.axis('tight')

	'''
	scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),1)
	
	# Draw heatmap of the validation accuracy as a function of gamma and C
	#
	# The score are encoded as colors with the hot colormap which varies from dark
	# red to bright yellow. As the most interesting scores are all located in the
	# 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
	# as to make it easier to visualize the small variations of score values in the
	# interesting range while not brutally collapsing all the low score values to
	# the same color.

	plt.figure(figsize=(8, 6))
	plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
	plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
	           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
	plt.xlabel('lol')
	plt.ylabel('C')
	plt.colorbar()
	plt.yticks(np.arange(len(C_range)), C_range)
	plt.title('Validation accuracy')
	plt.show()

# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

