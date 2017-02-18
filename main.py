# Calvin Wang & Tate Allen

import numpy as np
from numpy import *
import pandas as pd # Data frames
import matplotlib.pyplot as plt # Visuals
import seaborn as sns # Danker visuals
import tensorflow as tf #swaaaaaaaag
import tempfile 
from sklearn.model_selection import train_test_split # Create training and test sets
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.neural_network import MLPClassifier # Neural Networks
from sklearn.metrics import roc_curve # ROC Curves

pd.set_option('display.max_columns', 500)
plt.style.use('ggplot')

names = ['id_number', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
         'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
         'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
         'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
         'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
         'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']


DATA_PATH = '/Users/calvin-is-seksy/Desktop/breastCancerClassification/breast-cancer-wisconsin.csv'

breastCancerDF = pd.read_csv(DATA_PATH, names = names)

breastCancerDF.set_index(['id_number'], inplace = True)
breastCancerDF.head()

breastCancerDF['diagnosis'] = breastCancerDF['diagnosis'].map({'M':1, 'B':0})

print("Here's the dimensions of our data frame:")
print(breastCancerDF.shape)
print("Here's the data types of our columns:")
print(breastCancerDF.dtypes)

breastCancerDF.describe()

# Variables chosen from Decision Trees modeling
breastCancerSamp = breastCancerDF.loc[:, ['concave_points_worst', 'concavity_mean', 'perimeter_worst', 'radius_worst', 'area_worst', 'diagnosis']]

sns.set_palette(palette = ('Blue', 'Red'))
pairPlots = sns.pairplot(breastCancerSamp, hue = 'diagnosis')
pairPlots.set(facecolor = '#fafafa')
plt.show()

# Pearson Correlation Matrix - to measure amount of correlation between every variable
corr = breastCancerDF.corr(method = 'pearson')

f, ax = plt.subplots(figsize = (11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 275, as_cmap = True)

# Draw the heatmap with the mask and correct aspect ratio
PCM = sns.heatmap(corr,  cmap=cmap,square=True,
            xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.yticks(rotation = 0)
plt.xticks(rotation = 270)
plt.show()

# Box Plot
f, ax = plt.subplots(figsize = (11, 15))

ax.set_facecolor('#fafafa')
ax.set(xlim = (-0.5, 50))
ax = sns.boxplot(data = breastCancerDF, orient = 'h', palette = 'Set2')
plt.show()

# Now let's try normalizing our data!
breastCancerTest = breastCancerDF.iloc[:, 1:]

for item in breastCancerDF:
	if item in breastCancerTest:
		normalizedBC = ((breastCancerTest - breastCancerTest.min()) /
						(breastCancerTest.max() - breastCancerTest.min()))

print(normalizedBC.iloc[:, 0:6].head())

# Now let's standardize our data!
breastCancerTest = (breastCancerTest - mean(breastCancerTest)) / std(breastCancerTest)
# unsure about accuracy :/

# Now let's concatenate a data frame
df = breastCancerDF.drop(breastCancerTest, axis = 1)
breastCancerDF = pd.concat([normalizedBC, df], axis = 1)

print(breastCancerDF.shape)
print("Let's view some of our variables:")
print(breastCancerDF.iloc[:, 0:5].head())

# Now let's see a box plot of our normalized data
f, ax = plt.subplots(figsize = (11, 15))

ax.set_facecolor('#fafafa')
ax.set(xlim=(-0.5, 1.05))
ax = sns.boxplot(data = breastCancerDF[1:29], orient = 'h', palette = 'Set2') 
plt.show() 

# Now that we have cleaned up our data, let's set up our training set & test set 
train, test = train_test_split(breastCancerDF, test_size = 0.2, random_state = 42) 

train_set = train.ix[:, train.columns != 'diagnosis'] 
class_set = train.ix[:, train.columns == 'diagnosis'] 

test_set = test.ix[:, test.columns != 'diagnosis']
test_class_set = test.ix[:, test.columns == 'diagnosis']

np.save('normalizedData/trainingSetX.npy', train_set) 
np.save('normalizedData/trainingSetY.npy', class_set) 

np.save('normalizedData/testSetX.npy', test_set) 
np.save('normalizedData/testSetY.npy', test_class_set) 

# Let's begin training some learning algorithms! 
# Start with a relatively simple one: linear regression with tensorflow 
inputs = tf.placeholder(tf.float32, shape=[None, 30])
labels = tf.placeholder(tf.float32, shape=[None, 2])

Wout = tf.Variable(tf.truncated_normal([30, 2], stddev=0.1))
bout = tf.Variable(tf.constant(0.1, shape=[2]))
output_layer = tf.nn.xw_plus_b(inputs, Wout, bout)


















