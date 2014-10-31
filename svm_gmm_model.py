__author__ = 'cboys'

## This code runs a support vector machine model using mixed Gaussian
## methods on the Kaggle competition data for scikit-learn tutorials.
## This code is not mine and was taken from the forums, where it was
## written down by Luan Junyi. All I have done is annotated the code
## and changed a few details and altered some commands for
## compatibility with my idiosynchratic python install.
##


# Read in the data from the Kaggle competition


# We reduce the dimension of the data to two using principal
# component analysis. Whitening is a standard data-cleaning
# process that divides input by its variance to reduce noise
# and inter-correlation.

# Use scikit-learn's PCA class to define a two-dimensional
# version of the data for plotting purposes, with whitening

# pca2 = PCA(n_components=2, whiten=True)

# Fit the PCA to the data (concatenting the train and test arrays)
# and apply the dimensionality reduction to the training data.

#pca2.fit(np.r_[X, X_test])
#X_pca = pca2.transform(X)

# Show a plot of everything; notice the data is not nicely separable
# so a simple linear SVC will not be effective.

# i0 = np.argwhere(y == 0)[:, 0]
# i1 = np.argwhere(y == 1)[:, 0]
# X0 = X_pca[i0, :]
# X1 = X_pca[i1, :]
# plt.plot(X0[:, 0], X0[:, 1], 'ro')
# plt.plot(X1[:, 0], X1[:, 1], 'b*')
# plt.show()

# Now let's see exactly how many principle components of the 40-
# dimensional data we actually need. Again, let's fit the PCA to
# the full array of test and training data. Since we have no labels
# for our variables, we can run this completely blind.

#Xpca = pca.fit_transform(X,trainLabels)
#print pca.explained_variance_ratio_

# var = 0
# for i in range(0,11):
#     var=var+pca.explained_variance_ratio_[i]
# print var

# As we can see, there is a great deal of variance explained by the
# first twelve principal components; all further PCs explain an order
# of magnitude less, and the final three just look like noise and so
# may be discarded.

# Our next step is to try and understand how each principal component
# is distributed. We use Gaussian kernel density estimation for this.
# We plot histograms for the PCs (or at least some of them) to show
# they look Gaussian.

# print X_all[:,0]
# plt.hist(X_all[:,0],bins=100)
# plt.show()
#
# plt.clf()

def qq_plot(x):
    from scipy.stats import probplot
    probplot(x, dist='norm', plot=plt)

# qq_plot(X[:,0])
# plt.show()

# See what we get just using Gaussians and the 12 PCs with SVM??

# Note that Gaussian is a very good fit for the data except at the very
# far ends. Let's use Gaussian mixed models and see if this works better.
