#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# In[75]:


# 데이터 분석, 아래 패키지들 다운로드 하려면 cmd창에서 pip install 패키지명
# test data에는 
import pandas as pd #데이터 분석을 하기위해 유용한 패키지, 데이터 분석/시각화/컬럼 하는데 많이 쓰이는 패키지 
#(프로젝트 수행 시 많이 사용 할 것같음)
import numpy as np #(프로젝트 수행 시 많이 사용 할 것같음)
import random as rnd #랜덤한 숫자 

# 시각화
import seaborn as sns #seaboarn은 matplotlib보다 화려한 시각화를 할 수 있는 패키지
import matplotlib.pyplot as plt #그래프를 그릴 수 있는 간단한 패키지
#get_ipython().run_line_magic('matplotlib', 'inline')
#주피터 노트북 안에서 실행해서 보여줄 수 있게 함

# 기계 학습
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC #svm는 support veter mucine. 숫자 예측에 쓰이는
#것은 SVR이라 하고 분류에 쓰이는 것은 SVC라고 많이 함. 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# ## 데이터 로딩 및 확인

# In[76]:


# 데이터 로딩, 
#절대경로 : 해당 디렉토리위치를 root디렉토리부터 /로 표현, #상대경로:현재 디렉토리 기준 경로를 .으로 표현, data폴더 생성 후 파일 넣음
train_df = pd.read_csv('./data/train.csv') #csv로 데이터 분할 후 dafa frame형(dandas제공타입, 데이타 부분을 쉽게 접근, 확인,연산 가능한 자료구조)
test_df = pd.read_csv('./data/test.csv') #train_df에 넣음
combine = [train_df, test_df] #두 개 파일을 합쳐서 combine에 넣음 


# In[77]:


type(train_df)
#데이터 타입을 확인
#pandas에서 제공하는 데이터 프레임이라는 기본 형식임. 


# In[78]:


print(train_df.columns.values) #train데이터 안에 colum을 보여줌. 자동적으로 colum 부분을 인식해서 보여줌 


# In[79]:


# preview the data, #kernel - restart하면 미리 나온 내용 삭제 가능 
train_df.head() #train df라는 데이터의 내용 중 상위 데이터 5개 까지만 보여줌


# In[80]:


# preview tail of the data
train_df.tail()


# In[81]:


train_df.info() #data에 대한 정보 표시, 데이터 타입에 대한 설명, 다각도로 봄 
print('_'*40)
test_df.info()


# In[82]:


train_df.describe() #살아남으면 1, 아니면0이므로 살아남은 사람이 많으면 1에 더 가까움 


# ## Missing Value 처리

# In[83]:


# check missing values in train dataset, 데이터 중에 없는 부분(수집 실수, 처리 중 소실 등) 처리
train_df.isnull().sum() #0이랑은 다름. 없으면 null이 들어감. null값이 있는지, 있다면 몇 개나 있는지 카운팅


# In[84]:


test_df.isnull().sum() #test데이터에서 missing data 확인


# ###### 속성에 값이 없는 샘플들이 존재. 그 중 Age, Cabin에 missing value(누락값)가 많이 발견됨 전체에서 몇프로나 차지하는 지 확인해보자.

# In[85]:


# Age에 누락값이 있는 샘플의 비율
sum(pd.isnull(train_df['Age']))/len(train_df["PassengerId"]) #train_df에서 Age를 pandas의 기능을 활용하여 isnull을 확인(true,false로나옴)
#train데이터 중 passengerId컬럼 해당이 len몇 명인지 (""안에 적으면 열 확인)


# In[86]:


len(train_df["PassengerId"])


# In[87]:


sum(pd.isnull(train_df['Age']))
#true는 1, false는 0이므로 sum을 하면 true인것만 sum이 됨 


# In[88]:


# Cabin에 누락값이 있는 샘플의 비율
sum(pd.isnull(train_df['Cabin']))/len(train_df["PassengerId"]) #77프로 샘플에서 누락이 됨. 


# ###### Age는 20%의 샘플에서 누락, Cabin은 77%의 샘플에서 누락. Age는 누락된 값을 채워넣고, Cabin은 아예 feature를 버리는게 낫겠다.

# In[89]:


# Age가 어떻게 분포되어 있는지 히스토그램으로 확인해보자.
ax = train_df["Age"].hist(bins=15, color='teal', alpha=0.8) #age 컬럼을 가지고 히스토그램을 만듦. 
ax.set(xlabel='Age', ylabel='Count')
plt.show()


# In[90]:


# 그럼 Age의 중간값은 얼마일까?

train_df["Age"].median(skipna=True) #중간값을 알려주는 것이 median임 , null값은 지워서 표시하는 것이 skipna= True


# ###### 누락된 값들에 그냥 중간값을 일괄적으로 채워넣는 것이 좋아보인다. 남은 것은 train에서 Embarked, test에서 Fare

# In[17]:


# train set에서 Embarked 의 분포를 확인해보자.
sns.countplot(x='Embarked',data=train_df,palette='Set2') #palette는 그래프 색상 지정(set1,2,3중에 지정 가능)
plt.show()


# In[18]:


# Fare는 가격, 그럼 평균값은?
train_df["Fare"].mean(skipna=True)


# ###### Embarked는 S가 가장 많으니 누락값에 S를 채워넣으면 무난하겠다.
# ###### Fare는 승선한 항구와 티켓 등급에 따라 다르겠지만 편이상 평균값인 32를 취해서 누락값에 넣는 것을 채택하겠다.

# ## 데이터 전처리: 속성에 따라 누락된 값을 채워 넣거나 속성 자체 제거

# In[19]:


# 누락된 값을 적절한 값으로 채워넣기하는 것이 fillna임 
train_df["Age"].fillna(28, inplace=True) 
test_df["Age"].fillna(28, inplace=True) 
train_df["Embarked"].fillna("S", inplace=True) 
test_df["Fare"].fillna(32, inplace=True) 

# 누락된 값이 너무 많은 속성 제거
train_df.drop('Cabin', axis=1, inplace=True) #drop 명령어를 사용해서 컬럼 제거 (row를 제거하기 위해서는 axis=0하면 됨)
test_df.drop('Cabin', axis=1, inplace=True)


# ## 데이터 속성별 값에 따라 생존자 확인
# 

# In[20]:


#데이터 속성별로 값에 따라 생존자 확률 (속성값에 따라 그룹핑)


# In[21]:


# 객실 등급에 따른 생존자 확률(0,1)로 구한 것이기 때문, 컬럼을 선택해서 표현
#Pclass로 그룹핑해서 survivied의 평균(mean)을 구해서 ascending=False이므로 
#큰 값부터 정렬해서 표시함. 
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[22]:


# 성별에 따른 생존자 확률, []로 묶어서 사용해야 함 
#여성 생존률이 더 높음 
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[23]:


# 가족, 친척의 명수에 따른 생존자 확률
#가족이나 친척이 한두명 있는 경우보다 생존률이 높다. 세명 이상인 경우는 드물기 
#때문에 적게 나왔을 수 있다.
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[24]:


# 부모자식 관계에 있는 사람의 명수에 따른 생존자 확률
#주모자식 관계에 있는 사람이 한명 이상인 경우 아예 없는 경우보다 생존률이 높다. 
#4명 이상인 경우는 드물기 때문에 값이 적게 나왔을 수 있다. 
#그럼 SibSp와 Parch속성을 Numeric이 아닌 Binary 값으로 대체하여 단지 가족이 있는지 없는지,
#부모자식 관계의 사람이 있는지 없는지만을 따질 수도 있겠다. 
#사람수가 많다고 살아남을 확률이 높은게 아니었기 때문에!
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ## 데이터 시각화

# In[25]:


# 나이별 비생존자와 생존자, sns는 seaborn이었음. 
g = sns.FacetGrid(train_df, col='Survived') # 열에 생존자 0/1 
g.map(plt.hist, 'Age', bins=20) #히스토그램(hist)응 그릴 거고 age 기준으로 볼것.
#단순히 20~30대가 사람이 많아서 많이 나온 것일 수도 있음. 
#상대적으로 얼마나 죽고 살았는지는 알 수 있음. 


# In[26]:


# 나이별 객실 등급별 비생존자와 생존자
#예측을 할 때 데이터가 어떤 것을 보여주고 있고 어떤것을 의미하고 있는지 파악하는 것도 중요
#결과 값을 가지고 더 효과적인 방법을 찾을 수도 있음. 따라서 의미있는 작업.
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6) # 열에 생존자, 행에 객실 등급 
grid.map(plt.hist, 'Age', alpha=.5, bins=20) #aplpa는 색상의 농도를 결정 
grid.add_legend();


# In[27]:


#이번에는 히스토그램이 아닌 꺾은선 그래프
#항구 이름에 따라, 남/여, pClass따라 생존자 확인 
# seaborn패키지 설명을 직접 찾아 어떤 그래프를 그릴 수 있는지 확인해볼것.(참고해서 프로젝트 적용)
grid = sns.FacetGrid(train_df, col='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order=[1, 2, 3], hue_order=None)
grid.add_legend()


# In[28]:


#이제 그래프까지 확인했으니 위에서 분석한 내용을 바탕으로 데이터 전처리 다시 수행 


# ## 데이터 전처리: 속성 조정

# In[29]:


# 우선 현재 보유하고 있는 속성을 다시 한 번 확인해보자
train_df.head()


# ###### 속성 조정
# 1. PassengerId는 샘플별로 다르기 때문에 제거(의미가 없음)
# 2. Survived 는 예측해야할 output
# 3. Age, Fare는 그대로 채택
# 4. Sex, Pclass, Embarked는 카테고리 값이므로 처리. (Sex와 같이 글자거나 카테고리가 여러개인 것 처리)
# 5. SibSp, Parch 는 Binary 값으로 수정
# 6. Ticket은 표 번호이므로 상관성이 거의 없는 값이라 제거
# 7. Name은 한 번 살펴볼 것.
# 
# ###### > 어떤 속성을 선택할 것인지 

# ### 데이터 전처리 : 속성 조정 - SibSp, Parch 는 Binary 값으로 수정

# In[30]:


# 신규 속성인 TravelSibSp, TravelParch 만들어줌(바이너리 값으로 바꿔주면서 신규컬럼으로 변경 )
#인덱스를 찾아 해당 속성이 0보다 크면 1, 아니면 0으로 새로운 컬럼에 값을 넣게 됨. 
#numpy기능 중 속성을 찾아서 속성의 조건을 보고 값을 대체
train_df['TravelSibSp'] = np.where(train_df['SibSp']>0, 1, 0) 
train_df['TravelParch'] = np.where(train_df['Parch']>0, 1, 0)
# 이후 SibSp, SibSp 제거
train_df.drop('SibSp', axis=1, inplace=True)
train_df.drop('Parch', axis=1, inplace=True)


# In[31]:


# test 데이터도 마찬가지로 적용
test_df['TravelSibSp'] = np.where(test_df['SibSp']>0, 1, 0)
test_df['TravelParch'] = np.where(test_df['Parch']>0, 1, 0)
test_df.drop('SibSp', axis=1, inplace=True)
test_df.drop('Parch', axis=1, inplace=True)


# In[32]:


train_df.head(20)


# ### 데이터 전처리 : 속성 조정 - 카테고리 속성인 Pclass, Embarked, Sex 처리
# 
# 어떻게? Pclass에 세 가지가 있으니 Pclass 라는 속성을 세 개로 쪼갠다. Pclass_1, Pclass_2, Pclass_3
# 
# Embarked도 마찬가지. S, C, Q 가 있으니 Embarked_S, Embarked_C, Embarked_Q
# 
# Sex도 마찬가지, female, male 이 있으니 Sex_female, Sex_femal

# In[33]:


train_df


# In[34]:


train_df2 = pd.get_dummies(train_df, columns=["Pclass"])
train_df2
#Plcass 안에있는 값들을 통해 새로운 컬럼을 만들어 줌 


# In[35]:


# Pcalss를 위한 새로운 카테고리 속성을 만들어 새롭게 저장 (train_df2)
#Pclass는 카테고리컬한 속성이기 때문에 속성을 쪼갬.
# 즉 Pclass는 3개였으므로 3개로 카테고리가 나윔. 
#get_dumies 는 카테고리컬한 속성을 다룸. 카테고리 속성(value)를 확인하여 해당 속성
#개수만큼 새로운 카테고리를 더미로 생성해 줌. 
train_df2 = pd.get_dummies(train_df, columns=["Pclass"])

# Embarked를 위한 새로운 카테고리 속성을 만들어 새롭게 저장 (train_df3)
# Embarked에 S,C,Q는 카테고리컬한 value임. 만약 단순하게 1,2,3으로 값을 변경해서 
#들어가게 되면 순서가 생겨버리기 때문에 새로운 카테고리 속성을 만들어 저장한는 것. 
#따라서 S,C,Q에 대한 카테고리를 각각 생성하여 해당 카테고리에 속하면 1,0 으로 바이너리로 표현해주는 것.  
#카테고리 개수만큼 새로운 컬럼이 생기는 것. 
train_df3=pd.get_dummies(train_df2, columns=["Embarked"])

# Sex를 위한 새로운 카테고리 속성을 만들어 새롭게 저장 (train_df4)
train_df4=pd.get_dummies(train_df3, columns=["Sex"])

# 결과 확인
train_df4.head()


# ###### 그런데 여기서, Sex_female, Sex_male 이 모두 필요할까? 어짜피 같은 정보를 갖고 있으므로 둘 중 하나만 있으면 되지 않는가? 따라서 둘 중 하나를 삭제
# 여성이 아니면 당연히 남성이기 때문에 의미하는 바가 같음.

# ### 데이터 전처리 : 속성 조정 - 쓸모없는 속성 제거

# In[36]:


train_df4.drop('PassengerId', axis=1, inplace=True)
train_df4.drop('Name', axis=1, inplace=True)
train_df4.drop('Ticket', axis=1, inplace=True)
train_df4.drop('Sex_male', axis=1, inplace=True)
train_df4.head()


# ### 데이터 전처리 : 위의 속성 조정을 이젠 test_df에도 모두 해주자

# In[37]:


test_df2 = pd.get_dummies(test_df, columns=["Pclass"])
test_df3 = pd.get_dummies(test_df2, columns=["Embarked"])
test_df4 = pd.get_dummies(test_df3, columns=["Sex"])

#test_df4.drop('PassengerId', axis=1, inplace=True) <--- 이건 나중에 평가를 위해 일단 지금은 지우지 말자
test_df4.drop('Name', axis=1, inplace=True)
test_df4.drop('Ticket', axis=1, inplace=True)
test_df4.drop('Sex_male', axis=1, inplace=True)
test_df4.head() #만약 test로 예측하고 싶다면 train 데이터를 쪼개서 사용 


# ###### 이제 드디어 데이터 준비는 모두 끝났다!!!!!!!!

# # Machine Learning 기법을 활용한 생존자 예측
# ## 활용 모델
# 
# Logistic Regression
# 
# k-Nearest Neighbors
# 
# Support Vector Machines
# 
# Naive Bayes classifier
# 
# Decision Tree
# 
# Artificial neural network

# In[38]:


# 우선 학습 집합과 테스트 집합을 준비한다.
#Survived는 예측을 해야하는 값이기 때문에 Y에 해당 됨. 따라서 쓰지 않음. 
X_train = train_df4.drop("Survived", axis=1)
Y_train = train_df4["Survived"]
X_test = test_df4.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# ## Support Vector Machines

# In[39]:


# SVM 모델 학습
svc = SVC()
svc.fit(X_train, Y_train)


# In[52]:


# 테스트 데이터에 대해 예측
Y_pred_svc = svc.predict(X_test)
# 테스트 데이터를 현재 레이블이 없으므로 학습 데이터에 대해 예측한 정확도 측정
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# ## Logistic Regression

# In[41]:


# Logistic Regression training. Linear모델 중 하나. 위에서 import 해놨음. 
#logistic 과 거의 흡사. 
logreg = LogisticRegression() #객체생성하면서 weight 를 저장할 수 있지만, 
#아직 학습 전이기 때문에 비어 있음. 
logreg.fit(X_train, Y_train) #이렇게 쓰면 학습이 되는 것. 


# In[48]:


# Logistic Regression prediction
Y_pred_logreg = logreg.predict(X_test) #X_test를 넣어서 test를 한 결과를 저장 
Y_pred_logreg


# In[49]:


acc_log = round(logreg.score(X_train, Y_train) * 100, 2)#traing 데이터에 대해 예측 정확도 구함
#train 데이터를 사용하는 이유는 test 데이터는 컨페티션 용이기 때문에 Y값 즉,
#survived에 대한 레이블이 없음. 따라서 train data 로 예측하는 것. 
acc_log


# In[51]:


# x_train을 넣어 예측했던 결과와 실제 값 y_train을 비교해서 얼마나 일치하는지 확인하는 것. 
logreg.score(X_train, Y_train)


# In[45]:


# 속성별 상관 계수
#예측하는 값(생존률)에 대한 상관관계이기 때문에 값이 높을수록 살아남은 확률 높은 것

coeff_df  =  pd.DataFrame(train_df4.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# ## k-Nearest Neighbor

# In[53]:


knn  = KNeighborsClassifier(n_neighbors = 3) #K의 개수를 정해 줌 
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# ## Naive Bayes classifiers

# In[54]:


# Gaussian Naive Bayes
gaussian = GaussianNB() #이전에 배웠던 나이브베이즈는 숫자로 값이 들어 감. 
#디스크립트한 속성이 아닌 뉴메릭, 컨티니우스한 속성이 될 수도 있기 때문에 이 때는 
#Gauusian naive bayes를 사용(실수인 경우). 여기선 컨티니우스 속성이기 때문에! 
gaussian.fit(X_train, Y_train)
Y_pred_NB = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# ## Decision tree

# In[56]:


# Decision Tree
#정확한 패턴이 있는 경우 룰에 의해 decisiontree 로 하는게 더 좋을 수 있음
#여자인 경우, 클래스가 높은 경우 등 패턴이 있었음
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_DT = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# ## Artificial Neural Network

# In[62]:


ANN = MLPClassifier(solver='lbfgs',alpha=1, hidden_layer_sizes=(10,20),random_state=1 ) 
#첫 번째 hidden layer 는 20, 2번째 hidden layer는 10개 준 것. (기준이 없기 때문에 값 정하기 어려움)
#solver는 옵티마이즈 하는 것
ANN.fit(X_train, Y_train)
Y_pred_ANN = ANN.predict(X_test)
acc_ANN = round(ANN.score(X_train, Y_train) * 100, 2)
acc_ANN


# ## 최종 결과 저장

# In[65]:


Y_pred = Y_pred_DT #decission tree가 결과가 제일 좋게 나왔었음 

submission = pd.DataFrame({ #dataframe을 새로 만듦. 
        "PassengerId": test_df4["PassengerId"], #각각 컬럼에 결과값을 넣어 새로운 data frame을 만든 것
        "Survived": Y_pred
    })
submission.to_csv('./data/submission.csv', index=False)


# In[66]:


submission


# ### Confusion Matrix

# In[72]:


submission.to_csv('./data/submission.csv', index=False) 
#csv, 엑셀처럼 데이터를 분할(구분) 해서 표현 


# In[73]:


#위에서 학습한 svc활용, training data에대한 예측값 저장
Y_pred_svc_train = svc.predict(X_train) #객체.predict(데이터)


# In[91]:


#sklearn에서 제공하는 confusion matrix
#confusion matrix 형태로 확인할 것 
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(Y_train, Y_pred_svc_train)
CM


# In[ ]:




