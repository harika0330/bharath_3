import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="white", color_codes=True)

iris = pd.read_csv("./Datasets/IRIS.csv")
iris.head()

iris["Species"].value_counts()

#Scatter Plot


sns.FacetGrid(iris, hue="Species",height=6).map(plt.scatter, "Petal.Length", "Sepal.Width").add_legend()

#Converting categorical varibales into numbers

flower_mapping = {'setosa': 0,'versicolor': 1,'virginica':2}
iris["Species"] = iris["Species"].map(flower_mapping)

iris.head()

#Preparing inputs and outputs

X=iris[['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']].values
y=iris[['Species']].values 

#Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X, y)
#Accuracy

model.score(X,y)

expected = y
predicted = model.predict(X)
predicted

#summarize the fit of the model

from sklearn import metrics
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


#Regularization

model = LogisticRegression(C=20,penalty='l2' )
model.fit(X,y)
model.score(X,y)


#Effect of Regularization on classification boundary

from sklearn import linear_model, datasets
import numpy as np

def Regularization_Logistic(Regu,type):

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:,:2]  # we only take the first two features.
    Y = iris.target

    h = .02  # step size in the mesh

    logreg = linear_model.LogisticRegression(C=Regu,penalty=type)

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    #plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

    expected = Y
    predicted = logreg.predict(X)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

from IPython.html import widgets
from IPython.html.widgets import interact
from IPython.display import display
i = interact(Regularization_Logistic, Regu=(1,10000),type=['l1','l2'])



