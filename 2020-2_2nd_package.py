#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 드라이브 연동
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


#0번
direc = '/content/drive/My Drive/패키지'

import os
import pandas
import pandas as pd

os.chdir(direc)

data = pd.read_csv("dat.csv",encoding='utf-8')


# In[ ]:


data = data.dropna()


# In[ ]:


len(data['class'])


# In[ ]:


data.head()


# In[ ]:


#1번

#1-1번
vec = []
for doc in data['article']:
  vec.append(doc) 


# In[ ]:


vec[0]


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer 

counter = CountVectorizer(binary=True)
print(counter.fit(vec))


# In[ ]:


countdata = counter.transform(vec).toarray()
countdata[0]


# In[ ]:


#1-2번
from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf = TfidfVectorizer(binary=True)
print(tfidf.fit(vec))


# In[ ]:


tfidata = tfidf.transform(vec).toarray()
tfidata


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


# In[ ]:


import seaborn as sns, numpy as np
classdata = np.array(data['class'])
encoder = LabelEncoder()
encoder.fit(classdata)
labels = encoder.transform(classdata)
#labels = to_categorical(labels,num_classes=None)
print('인코딩 변환값:',labels)

#label categorical로바꿔라


# In[ ]:


labels


# In[ ]:


#1-3번
from sklearn.model_selection import train_test_split

cx_train, cx_test,y_train,y_test = train_test_split(countdata,labels ,test_size=0.2, shuffle=True,  random_state=1004)
tx_train, tx_test,y_train,y_test = train_test_split(tfidata,labels, test_size=0.2, shuffle=True,  random_state=1004)


# In[ ]:


y_train


# In[ ]:


len(y_train)


# In[ ]:


#2-1번_수정하기비율로 _accuracy
import matplotlib.pyplot as plt
import seaborn as sns, numpy as np
n, bins, patches = plt.hist(data['class'], bins=10,color = '#ffa299')
plt.show()


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
# 최적 alpha를 찾기 위한 grid search
alphas = np.arange(0.01, 1,0.02)
#count
x_train, x_valid,Y_train,Y_valid = train_test_split(cx_train,y_train ,test_size=0.25, shuffle=True,  random_state=1004)


# In[ ]:


#tfi
X_train, X_valid,Y2_train,Y2_valid = train_test_split(tx_train,y_train ,test_size=0.25, shuffle=True,  random_state=1004)


# 3-1

# In[ ]:


count_set = []

for alpha in alphas:
  count_grid = MultinomialNB(alpha=alpha)
  count_grid.fit(x_train, Y_train)
  pred_count_grid = count_grid.predict(x_valid)
  f1_count = metrics.f1_score(Y_valid, pred_count_grid, average='macro')

  count_set.append(f1_count)


# In[ ]:


best_alpha = alphas[np.argmax(count_set)]
best_alpha 


# In[ ]:


graph= plt.plot(alphas,count_set)


# In[ ]:


#tfi 
X_train, X_valid,Y2_train,Y2_valid = train_test_split(tx_train,y_train ,test_size=0.25, shuffle=True,  random_state=1004)


alphas = np.arange(0.01, 1,0.02)
tfi_set = []

for alpha in alphas:
  tfi_grid = MultinomialNB(alpha=alpha)
  tfi_grid.fit(X_train, Y2_train)
  pred_tfi_grid = tfi_grid.predict(X_valid)
  f1_tfi = metrics.f1_score(Y2_valid, pred_tfi_grid, average='macro')

  tfi_set.append(f1_tfi)


# In[ ]:


best_alpha = alphas[np.argmax(tfi_set)]
best_alpha 


# In[ ]:


graph= plt.plot(alphas,tfi_set)


# 3-2

# In[ ]:


from sklearn.svm import SVC 
# 최적 kernel를 찾기 위한 grid search
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

#count
#cx_train, cx_test,y_train,y_test = train_test_split(countdata,labels ,test_size=0.2, shuffle=True,  random_state=1004)
#x_train, x_valid,Y_train,Y_valid = train_test_split(cx_train,y_train ,test_size=0.25, shuffle=True,  random_state=1004)


f1_svm_set = []
for kernel in kernels:
  svm_grid = SVC(kernel=kernel)
  svm_grid.fit(x_train, Y_train)
  pred_svm_grid = svm_grid.predict(x_valid)
  f1_svm = metrics.f1_score(Y_valid, pred_svm_grid, average='macro')

  f1_svm_set.append(f1_svm)


# In[ ]:


best_kernel = kernels[np.argmax(f1_svm_set)]
best_kernel


# In[ ]:


clf_svm = SVC(kernel=kernel)
clf_svm.fit(x_train, Y_train)


# In[ ]:


pred_svm = clf_svm.predict(cx_test)

a = metrics.f1_score(y_test, pred_svm, average='macro')
a


# In[ ]:


#tfi
#tx_train, tx_test,y_train,y_test = train_test_split(tfidata,labels, test_size=0.2, shuffle=True,  random_state=1004)
#X_train, X_valid,Y2_train,Y2_valid = train_test_split(tx_train,y_train ,test_size=0.25, shuffle=True,  random_state=1004)
from sklearn.svm import SVC 
# 최적 kernel를 찾기 위한 grid search
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

f1_svm_set = []
for kernel in kernels:
  svm_grid = SVC(kernel=kernel)
  svm_grid.fit(X_train, Y2_train)
  pred_svm_grid = svm_grid.predict(X_valid)
  f1_svm = metrics.f1_score(Y2_valid, pred_svm_grid, average='macro')

  f1_svm_set.append(f1_svm)


# In[ ]:


best_kernel = kernels[np.argmax(f1_svm_set)]
best_kernel


# In[ ]:


f1_svm_set[0]


# In[ ]:


clf_svm = SVC(kernel=kernel)
clf_svm.fit(X_train, Y2_train)


# In[ ]:


pred_svm = clf_svm.predict(tx_test)
a = metrics.f1_score(y_test, pred_svm, average='macro')
a


# 3-3

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
#count
# 최적 alpha를 찾기 위한 grid search

#cx_train, cx_test,y_train,y_test = train_test_split(countdata,labels ,test_size=0.2, shuffle=True,  random_state=1004)
#x_train, x_valid,Y_train,Y_valid = train_test_split(cx_train,y_train ,test_size=0.25, shuffle=True,  random_state=1004)

k_s = range(1, 5)

f1_knn_set = []
for k in k_s:
  knn_grid = KNeighborsClassifier(n_neighbors=k) #parameter
  knn_grid.fit(x_train,Y_train)
  pred_knn_grid = knn_grid.predict(x_valid)
  f1_knn = metrics.f1_score(Y_valid, pred_knn_grid, average='macro')

  f1_knn_set.append(f1_knn)


# In[ ]:


best_k = k_s[np.argmax(f1_knn_set)]
best_k 


# In[ ]:


graph= plt.plot(k_s,f1_knn_set) 
#이를 통해 k=3일 떄 가장 높은 정확도를 보임을 알 수 있다. (1일 떄 높은 이유 : 과대 적합)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, Y_train)


# In[ ]:


pred_knn = knn.predict(cx_test)
a = metrics.f1_score(y_test, pred_knn, average='macro')
a


# In[ ]:


#tfi
#tx_train, tx_test,y_train,y_test = train_test_split(tfidata,labels, test_size=0.2, shuffle=True,  random_state=1004)
#X_train, X_valid,Y2_train,Y2_valid = train_test_split(tx_train,y_train ,test_size=0.25, shuffle=True,  random_state=1004)
k_s = range(1, 5)
from sklearn.neighbors import KNeighborsClassifier
f1_knn_set = []
for k in k_s:
  knn_grid = KNeighborsClassifier(n_neighbors=k)
  knn_grid.fit(X_train,Y2_train)
  pred_knn_grid = knn_grid.predict(X_valid)
  f1_knn = metrics.f1_score(Y2_valid, pred_knn_grid, average='macro')

  f1_knn_set.append(f1_knn)


# In[ ]:


best_k = k_s[np.argmax(f1_knn_set)]
best_k


# In[ ]:


graph= plt.plot(k_s,f1_knn_set)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y2_train)


# In[ ]:


pred_knn = knn.predict(tx_test)
a= metrics.f1_score(y_test, pred_knn, average='macro')
a


# 3-4

# In[ ]:


svm과 나이브베이즈분류모델이 비슷하게 f1 score 값이 1에 비슷하기에 두개가 비슷하다고 볼 수 있다. 

