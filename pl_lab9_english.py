#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[9]:


df = pd.read_csv("titanic.csv")


# In[11]:


df.head()


# In[12]:


numeric_features = ['age', 'fare']
categorical_features = ['pclass', 'sex', 'embarked']


# In[13]:


X = df[numeric_features + categorical_features].copy()
y = df['survived'].astype(int) # Target: whether the passenger survived (0 or 1)


# In[28]:


X['useless_column'] = 'Taka Sama Wartosc'
categorical_features.append('useless_column')


# In[29]:


X.head()


# In[30]:


X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


X_tr.info()


# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin


# In[32]:


class DelOneValueFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.one_value_features = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.one_value_features = [col for col in X.columns if X[col].nunique() == 1]
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not self.one_value_features:
            return X
        return X.drop(columns=self.one_value_features)


# In[33]:


class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.classes_ = None

    def fit(self, X, y):
        # Store the unique classes from the target vector
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        np.random.seed(self.random_state)
        # Randomly assign classes for each row in X
        return np.random.choice(self.classes_, size=len(X))

    def predict_proba(self, X):
        np.random.seed(self.random_state)
        # Generate random probabilities that sum to 1
        raw_probs = np.random.rand(len(X), len(self.classes_))
        return raw_probs / raw_probs.sum(axis=1, keepdims=True)


# In[18]:


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


# In[34]:


categorical_transformer = Pipeline(steps=[
    ("remover", DelOneValueFeature()),  # Remove 'useless_column' before OneHotEncoder!
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


# In[20]:


preprocessor = ColumnTransformer(transformers=[
    ("num_trans", numeric_transformer, numeric_features),
    ("cat_trans", categorical_transformer, categorical_features)
])


# In[21]:


full_pipeline = Pipeline(steps=[
    ("preproc", preprocessor),
    ("model", LogisticRegression())
])


# In[35]:


param_grid = [
    # Grid 1: Random Forest
    {
        "preproc__num_trans__imputer__strategy": ["mean", "median"],
        "model": [RandomForestClassifier(random_state=42)],
        "model__n_estimators": [10, 50, 100],
        "model__max_depth": [None, 5, 10]
    },
    # Grid 2: Logistic Regression
    {
        "preproc__num_trans__imputer__strategy": ["mean", "median"],
        "model": [LogisticRegression(max_iter=1000)],
        "model__C": [0.1, 1.0, 10.0]
    },
    {
        "model": [RandomClassifier(random_state=42)]
    }
]


# In[23]:


from sklearn.model_selection import GridSearchCV


# In[36]:


grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, verbose=1, n_jobs=-1)


# In[37]:


grid_search.fit(X_tr, y_tr)


# In[38]:


print("\n" + "="*40)
print("NAJLEPSZE PARAMETRY:")
print("="*40)
for param, val in grid_search.best_params_.items():
    print(f"{param}: {val}")

# Performance on the test set for the best model
best_model_score = grid_search.score(X_test, y_test)


# In[40]:


best_model_score


# In[41]:


print("\n" + "="*40)
print("NAJLEPSZE PARAMETRY:")
print("="*40)
for param, val in grid_search.best_params_.items():
    print(f"{param}: {val}")

# Performance on the test set for the best model
best_model_score = grid_search.score(X_test, y_test)

# Let's check how a purely random model would perform for comparison
random_baseline = RandomClassifier(random_state=42)
random_baseline.fit(X_tr, y_tr)
random_score = random_baseline.score(X_test, y_test)

print("\n" + "="*40)
print("ACCURACY COMPARISON:")
print("="*40)
print(f"Nasz najlepszy model z GridSearch: {best_model_score:.4f}")
print(f"Model stricte losowy (Baseline):  {random_score:.4f}")


# ## przypomnienie - dane ustruktyryzowane
# 
# Na poprzednich zajęciach omawialiśmy wykorzystanie modelu regresji liniowej dla danych ustrukturyzowanych. 
# W najprostszym przypadku dla jednej zmiennej X i jednej zmiennej celu moglibyśmy np. przypisać model w postaci: 
# 
# satysfakcja_z_zycia = $\alpha_0$ + $\alpha_1$ PKB_per_capita 
# 
# $\alpha_0$ nazywamy punktem odcięcia (_intercept_) albo punktem obciążenia (_bias_)

# In[42]:


import numpy as np

np.random.seed(42) 
m = 100
X = 2*np.random.rand(m,1) 
a_0, a_1 = 4, 3
y = a_0 + a_1 * X + np.random.randn(m,1)


# In[45]:


import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.show()


# W ogólności model liniowy: 
# $\hat{y} = \alpha_0 + \alpha_1 x_1 + \alpha_2 x_2 + \dots + \alpha_n x_n$
# gdzie $\hat{y}$ to predykcja naszego modelu (wartość prognozowana), dla $n$ cech przy wartościach cechy $x_i$. 
# 
# W postaci zwektoryzowanej możemy napisać: 
# $\hat{y} = \vec{\alpha}^{T} \vec{x}$
# 
# W tej postaci widać dlaczego w tym modelu dokłada się kolumnę jedynek - wynikają one z wartości $x_0$ dla $\alpha_0$. 

# In[46]:


# dodajmy jedynkę do naszej tabeli 
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)


# Powiedzieliśmy, że możemy w tym modelu znaleźć funkcję kosztu 
# 
# $MSE(\vec{x}, \hat{y}) = \sum_{i=1}^{m} \left( \vec{\alpha}^{T} \vec{x}^{(i)} - y^{(i)} \right)^{2}$
# 
# Tak naprawdę możemy $MSE(\vec{x}, \hat{y}) = MSE(\vec{\alpha})$
# 
# Rozwiązanie analityczne: 
# $\vec{\alpha} = (X^{T}X)^{-1} X^T y$

# In[47]:


alpha_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y


# In[48]:


alpha_best, np.array([4,3])


# In[49]:


X_new = np.array([[0],[2]])


# In[50]:


X_new_b = add_dummy_feature(X_new)


# In[51]:


y_predict = X_new_b @ alpha_best


# In[52]:


import matplotlib.pyplot as plt

plt.plot(X_new, y_predict, "r-", label="prediction")
plt.plot(X,y, "b.")
plt.show()


# In[53]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y) 

print(f"a_0={lin_reg.intercept_[0]}, a_1 = {lin_reg.coef_[0][0]}")

print("predykcja", lin_reg.predict(X_new))


# In[54]:


# Logistic Regression w scikit learn oparta jest o metodę lstsq 
alpha_best_svd, _, _, _ = np.linalg.lstsq(X_b, y, rcond=1e-6)
alpha_best_svd


# ## Gradient prosty
# 
# Pamiętaj o standaryzacji zmiennych (aby były one reprezentowane w tej samej skali). 
# 
# ### Wsadowy gradient prosty
# 
# W celu implementacji musimy policzyć pochodne cząstkowe dla funkcji kosztu wobec każdego parametru $\alpha_i$.
# 
# $\frac{\partial}{\partial \alpha_j}MSE(\vec{x}, \hat{y}) = 2 \sum_{i=1}^{m} \left( \vec{\alpha}^{T} \vec{x}^{(i)} - y^{(i)} \right) x_j^{(i)}$
# 
# Komputery posiadają własność mnożenia macierzy co pozwala obliczyć nam wszystkie pochodne w jednym obliczeniu. Wzór i algorytm liczący wszystkie pochodne "na raz" wykorzystuje cały zbiór X dlatego też nazywamy go **wsadowym**.
# 
# Po obliczeniu gradientu po prostu idziemy "w przeciwną stronę" 
# 
# $ \alpha_{next} = \alpha - \eta \nabla_{\alpha} MSE(\alpha)$

# In[55]:


from IPython.display import Image


# In[57]:


Image(filename='02_10.png', width=500)


# In[60]:


eta = 0.1
n_epochs = 1000
m = len(X_b)
np.random.seed(42) 
alpha = np.random.randn(2,1) # randomly initialize the solution
print(f"alpha init {alpha}")
for epoch in range(n_epochs):
    gradients = 2/m* X_b.T @ (X_b @ alpha - y)
    #print(alpha)
    alpha = alpha - eta*gradients


# In[61]:


alpha


# ## Stochastic gradient descent
# Jednym z poważniejszych problemów wsadowego gradientu jest jego zależność od wykorzystania (w każdym kroku) całej macierzy danych. Korzystając z własności statystycznych możemy zobaczyć jak będzie realizowała się zbieżność rozwiązania jeśli za każdym razem wylosujemy próbkę danych i na niej określimy gradient. Ze względu, iż w pamięci przechowujemy tylko pewną porcję danych algorytm ten może być używany dla bardzo dużych zbiorów danych. Warto jednak mieć świadomość, że tak otrzymane wyniki mają charakter chaotyczny, co oznacza, że funkcja kosztu nie zbiega się w kierunku minimum lecz przeskakuje dążąc do minimun w sensie średniej. 
# 

# In[62]:


n_epochs = 50
m = len(X_b)


def learning_schedule(t, t0=5, t1=50):
    return t0/(t+t1)

np.random.seed(42)
alpha = np.random.randn(2,1)

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1] 
        gradients = 2 * xi.T @ (xi @ alpha - yi)
        eta = learning_schedule(epoch * m + iteration) 
        alpha = alpha - eta * gradients


# In[63]:


alpha


# In[64]:


from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, 
                       penalty=None, eta0=0.01, 
                       n_iter_no_change=100, random_state=42)

sgd_reg.fit(X, y.ravel())


# In[65]:


sgd_reg.intercept_, sgd_reg.coef_


# In[70]:


from random import randint
randint(1,6)


# In[71]:


from random import randint

class Kosc():
    """ description """
    def __init__(self, sciany=6):
        """
        this method is executed during object initialization
        params: 
        number of sides (int)

        """
        self.sciany = sciany

    def roll(self):
        """ description"""
        return randint(1, self.sciany)


# In[74]:


a = Kosc(sciany=12)
[a.roll() for _ in range(10)]


# In[76]:


from random import choice
choice([0,1,2,3,4])


# In[77]:


from random import choice

class RandomWalk():
    def __init__(self, num_points=5000):
        self.num_points = num_points
        self.x_values = [0]
        self.y_values = [0]

    def fill_walk(self):
        while len(self.x_values) < self.num_points:
            # move left-right 
            # randomly choose a positive or negative direction and a distance 0-5 and assign them to variables

            x_direction = choice([-1,1])
            x_distance = choice([0,1,2,3,4])
            x_step = x_direction*x_distance

            y_direction = choice([-1,1])
            y_distance = choice([0,1,2,3,4])
            y_step = y_direction*y_distance

            # write a condition that skips the step when x_step and y_step are both 0 (use continue)
            if x_step == 0 and y_step == 0:
                continue

            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step

            self.x_values.append(next_x)
            self.y_values.append(next_y)


# In[84]:


rw = RandomWalk(10000)
rw.fill_walk()

rw.x_values[:5]


# In[85]:


import matplotlib.pyplot as  plt
point_number = list(range(rw.num_points))
plt.scatter(rw.x_values, rw.y_values, c=point_number, cmap=plt.cm.Blues,
           edgecolor='none', s=15)
plt.scatter(0,0,c='green', edgecolor='none', s=100)
plt.scatter(rw.x_values[-1], rw.y_values[-1],c='red', edgecolor='none', s=100)

plt.show()


# In[86]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])


# In[87]:


X = df.iloc[:100,[0,2]].values
y = df.iloc[0:100,4].values
y = np.where(y == 0, -1, 1)

import matplotlib.pyplot as plt


# In[88]:


plt.scatter(X[:50,0],X[:50,1],color='red', marker='o',label='setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue', marker='x',label='versicolor')
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.legend(loc='upper left')
plt.show()


# dziecko = Perceptron()
# dziecko.fit()
# 
# # the perceptron must have a learning-rate parameter
# dziecko.eta
# 
# # we can check how quickly it learns == how many errors it makes
# 
# dziecko.errors_ 
# 
# # the solution will be stored in the weights
# dziecko.w_
# # in our case the perceptron learns two weights!

# In[89]:


Image(filename='02_01.png', width=800) 


# In[90]:


Image(filename='02_04.png', width=800)


# In[91]:


Image(filename='02_02.png', width=800) 


# In[ ]:


class Perceptron():
    def __init__(self, n_iter=10, eta=0.01):
        self.n_iter = n_iter
        self.eta = eta

    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            pass
        return self


# In[92]:


import random

class Perceptron():

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):

        #self.w_ = np.zeros(1+X.shape[1])
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1+X.shape[1])] 
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                #print(xi, target)
                update = self.eta*(target-self.predict(xi))
                #print(update)
                self.w_[1:] += update*xi
                self.w_[0] += update
                #print(self.w_)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X)>=0.0, 1, -1)


# In[95]:


ppn = Perceptron()
ppn.fit(X,y)


# In[96]:


print(ppn.errors_)
print(ppn.w_)


# In[97]:


ppn.predict(np.array([-3, 5]))


# In[98]:


from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier, resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min() -1, X[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl,0], y=X[y==cl,1], alpha=0.8, color=cmap(idx), marker=markers[idx], label=cl)


# In[99]:


plot_decision_regions(X,y,classifier=ppn)
plt.xlabel("dlugosc dzialki [cm]")
plt.ylabel("dlugosc platka [cm]")
plt.legend(loc='upper left')
plt.show()


# In[ ]:




