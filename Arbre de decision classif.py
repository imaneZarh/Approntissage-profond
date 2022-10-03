#!/usr/bin/env python
# coding: utf-8

# In[27]:


from sklearn.datasets import load_iris
import numpy as np


# In[28]:


iris = load_iris()
X, y = iris.data, iris.target


# In[29]:


iris.feature_names


# In[30]:


X


# In[31]:


y


# In[32]:


#calcul de la moyenne
somme = X[0]
for i in range(1,150):
    somme = somme + X[i]
mean = somme/150


# In[33]:


mean


# In[34]:


#chaque liste represente une colonne
list_sepal_length=list()
list_sepal_width=list()
list_petal_length=list()
list_petal_width=list()
for row in X:
    list_sepal_length.append(row[0])
    list_sepal_width.append(row[1])
    list_petal_length.append(row[2])
    list_petal_width.append(row[3])


# In[35]:


#calcul de la moyenne et de l'ecart type
sepal_length_mean=np.mean(list_sepal_length)
sepal_length_ecart=np.std(list_sepal_length)

sepal_width_mean=np.mean(list_sepal_width)
sepal_width_ecart=np.std(list_sepal_width)

petal_length_ecart=np.std(list_petal_length)
petal_length_mean=np.mean(list_petal_width)

petal_width_ecart=np.std(list_petal_width)
petal_width_mean=np.mean(list_petal_width)


# In[42]:


print("sepal_length_mean = {}\nsepal_length_ecart = {} \n\nsepal_width_mean = {} \nsepal_width_ecart = {} \n\npetal_length_mean = {} \npetal_length_mean = {} \n\npetal_width_ecart = {} \npetal_width_mean = {} \n\n".format(sepal_length_mean,sepal_length_ecart,sepal_width_mean,
                                        sepal_width_ecart,petal_length_mean,petal_length_mean,
                                        petal_width_ecart,petal_width_mean))


# In[47]:


#nombre d'elements dans chaque classe
classe1=np.count_nonzero(y==0)
classe2=np.count_nonzero(y==1)
classe3=np.count_nonzero(y==2)

print("classe1 = {}\nclasse2 = {}\nclasse3 = {}\n".format(classe1,classe2,classe3))


# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
random_state=0)


# In[55]:


print(X_train[:, 0].size)
print(X_test[:, 0].size)


# In[58]:


print(y_train.size)
print(y_test.size)


# In[110]:


#construire l'arbre avec min_samples_leaf=20
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_leaf=20)
clf.fit(X_train, y_train)


# In[111]:


#Visualiser l'arbre
from matplotlib import pyplot as plt
tree.plot_tree(clf, filled=True)


# In[112]:


#prediction
predict = clf.predict(X_test)
predict


# In[113]:


#score
score = clf.score(X_test, y_test)
score


# In[114]:


#5% training 95% test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05,
random_state=0)


# In[ ]:





# In[116]:


#grid search
from sklearn.model_selection import GridSearchCV

values_depth = [1, 3, 5, 10, 20, 30, 40, 50] 
values_leaf=[20,30,50,100]
parameters={'max_depth':values_depth,'min_samples_leaf':values_leaf}
model = tree.DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=model,param_grid=parameters,verbose=True,cv=4)
grid_search.fit(X_train, y_train)


# In[122]:


#Meilleurs parametres a choisir
grid_search.best_params_


# In[123]:


#Meilleur score obtenue pour ces params
grid_search.best_score_


# In[142]:


import numpy as np
import matplotlib.pyplot as plt
# Paramètres
n_classes = 3
plot_colors = "bwy" # blue-white-yellow
plot_step = 0.02
# Choisir les attributs longueur et largeur des pétales
pair = [1, 2]
# On ne garde seulement les deux attributs
X = iris.data[:, pair]
y = iris.target
# Apprentissage de l'arbre
clf = tree.DecisionTreeClassifier().fit(X, y)
# Affichage de la surface de décision
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min,
y_max, plot_step))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.xlabel(iris.feature_names[pair[0]])
plt.ylabel(iris.feature_names[pair[1]])
plt.axis("tight")
# Affichage des points d'apprentissage
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
    cmap=plt.cm.Paired)
plt.axis("tight")
plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend()
plt.show()


# In[ ]:




