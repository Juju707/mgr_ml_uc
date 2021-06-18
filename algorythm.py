"""
Created on Sun May  9 17:31:10 2021

@author: Juju
"""

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

#Wczytanie danych
df_wzjg=pd.read_excel(r"C:\Users\Juju\Desktop\Test\WZJG.xlsx")
df_wzjg = df_wzjg.fillna(value=np.nan)

df_polip=pd.read_excel(r"C:\Users\Juju\Desktop\Test\Polipy 3.xlsx")
df_polip = df_polip.fillna(value=np.nan)
df_polip=df_polip.replace(r'*',np.nan)


#Usunięcie mało zapełnionych kolumn
df = pd.concat([df_wzjg, df_polip], ignore_index=True)
limit=df.shape[0]*0.3
counted=df.count()
counted_rows=df.count(axis=1)
for i, v in counted.items():
    if v<limit:
        del df[i]
        del df_wzjg[i]
        del df_polip[i]

#Wypełnianie danych
for key, value in df_wzjg.iteritems():
    col_median=value.median()
    col_std=value.std()
    col_std=float("{:.2f}".format(col_std))
    unique_values=value.unique()
    cleanedList = sorted([int(x) for x in unique_values if str(x) != 'nan'])
    for index, item in value.iteritems():
        val=-1
        if math.isnan(item):
            if key == 'MOCZ - BADANIE OGÓLNE  PRZEJRZYSTOŚĆ':
                my_list = ['1'] * 90 + ['3']* 10
                random_num=random.choice(my_list)
                df_wzjg.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE PASMA ŚLUZU':
                my_list = ['0'] * 19 + ['1'] * 36 + ['2'] * 11+['3']*23+['4']*12
                random_num=random.choice(my_list)
                df_wzjg.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE  KREW':
                my_list = ['0'] * 65 + ['1'] * 23 + ['2'] * 4+['3']*1+['4']*5
                random_num=random.choice(my_list)
                df_wzjg.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE  LEUKOCYTY':
                my_list = ['0'] * 82 + ['1'] * 7 + ['2'] * 3+['3']*3+['4']*3
                random_num=random.choice(my_list)
                df_wzjg.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE  KETONY':
                my_list = ['0'] * 67 + ['1'] * 9 + ['2'] * 9+['3']*8+['4']*3+['5']*2
                random_num=random.choice(my_list)
                df_wzjg.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE  BILIRUBINA':
                my_list = ['0'] * 100
                random_num=random.choice(my_list)
                df_wzjg.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE  AZOTYNY':
                my_list = ['0'] * 95  + ['2'] * 5
                random_num=random.choice(my_list)
                df_wzjg.loc[index,key]=float(random_num)
            else:
                while val<0:
                    val=col_median+ random.uniform(-col_std,col_std)
                df_wzjg.loc[index,key]=float("{:.2f}".format(val))
            

for key, value in df_polip.iteritems():
    col_median=value.median()
    col_std=value.std()
    col_std=float("{:.2f}".format(col_std))
    unique_values=value.unique()
    cleanedList = sorted([int(x) for x in unique_values if str(x) != 'nan'])
    for index, item in value.iteritems():
        val=-1
        if math.isnan(item):
            if key == 'MOCZ - BADANIE OGÓLNE  PRZEJRZYSTOŚĆ':
                my_list = ['0'] * 90  + ['2']* 10
                random_num=random.choice(my_list)
                df_polip.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE PASMA ŚLUZU':
                my_list = ['0'] * 24 + ['1'] * 48 + ['2'] * 13+['4']*14
                random_num=random.choice(my_list)
                df_polip.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE  KREW':
                my_list = ['0'] * 82 + ['1'] * 19 + ['2'] * 6+['3']*3
                random_num=random.choice(my_list)
                df_polip.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE  LEUKOCYTY':
                my_list = ['0'] * 74 + ['1'] * 12 + ['2'] * 5+['3']*3+['4']*6
                random_num=random.choice(my_list)
                df_polip.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE  KETONY':
                my_list = ['0'] * 64 + ['1'] * 14 + ['2'] * 8+['3']*9+['4']*2+['5']*3
                random_num=random.choice(my_list)
                df_polip.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE  BILIRUBINA':
                my_list = ['0'] * 98+['3']*1+['4']*1
                random_num=random.choice(my_list)
                df_polip.loc[index,key]=float(random_num)
            elif key == 'MOCZ - BADANIE OGÓLNE  AZOTYNY':
                my_list = ['0'] * 95  + ['2'] * 4 + ['3']*1
                random_num=random.choice(my_list)
                df_polip.loc[index,key]=float(random_num)
            else:
                while val<0:
                    val=col_median+ random.uniform(-col_std,col_std)
                df_polip.loc[index,key]=float("{:.2f}".format(val))

#x=df_polip["Wiek"].tolist()
#y=df_wzjg["Wiek"].tolist()             
#plt.hist([x,y], edgecolor='black', alpha=0.5, label=['Gr. kontrolna','Gr. badawcza'])
#plt.legend(loc='upper right')
#plt.xlabel('Wiek') 
#plt.ylabel('Ilosć pacjentów') 

#plt.show()

#Podział na zbiory
df = pd.concat([df_wzjg, df_polip], ignore_index=True)  
array = df.values
X=array[:,1:]
Y=array[:,0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size, random_state=seed, shuffle=True)          
sc=StandardScaler()
X_train_std=sc.fit_transform(X_train)
X_validation_std=sc.transform(X_validation)
cov_mat=np.cov(X_validation_std.T)
pca=PCA(n_components=12)
X_train_pca=pca.fit_transform(X_train_std,Y_train)
X_validation_pca=pca.fit_transform(X_validation_std)
#eigen_vals, eigen_vecs= np.linalg.eig(cov_mat)
#tot=sum(eigen_vals)
#var_exp=[(i/tot) for i in sorted (eigen_vals, reverse=True)]
#cum_var_exp=np.cumsum(var_exp)
#plt.bar(range(1,57), var_exp, alpha=0.5, align='center', label='Pojedyncza wariancja')
#plt.step(range (1,57), cum_var_exp, alpha=0.5, where='mid', label='Łączna wariancja')
#plt.ylabel('Współczynnik wariancji')
#plt.xlabel('Indeks głównej składowej')
#plt.legend(loc='best')
#plt.show()
#Prog 1%
df1=df[["Rozpoznanie choroby:","Wiek","MORFOLOGIA 5 DIFF NEUTROFILE#", "ŻELAZO", "MOCZ - BADANIE OGÓLNE NABŁONKI PŁASKIE", "FERRYTYNA", "WITAMINA B12", "KREATYNINA W SUROWICY  EGFR", "25-OH-WITAMINA D", "MOCZ - BADANIE OGÓLNE UROBILINOGEN","MORFOLOGIA 5 DIFF MPV","KWAS FOLIOWY","CRP ULTRACZUŁE", "MORFOLOGIA 5 DIFF MCH", "FOSFATAZA ALKALICZNA", "Kreatynina w surowicy.", "APTT", "MORFOLOGIA 5 DIFF MCHC", "MORFOLOGIA 5 DIFF HEMOGLOBINA","MORFOLOGIA 5 DIFF NIEDOJRZAŁE GRANULOCYTY # (meta-,mielo-,promielocyt)","MORFOLOGIA 5 DIFF BAZOFILE","PT (CZAS PROTROMBINOWY) WSKAŹNIK PROTROMBINY", "MORFOLOGIA 5 DIFF LIMFOCYTY#","MORFOLOGIA 5 DIFF MONOCYTY",   "Potas", "Bilirubina całkowita"]]
array = df1.values
X1=array[:,1:]
Y1=array[:,0]
validation_size = 0.20
seed = 7
X1_train, X1_validation, Y1_train, Y1_validation = model_selection.train_test_split(X1,Y1,test_size=validation_size, random_state=seed, shuffle=True)
X1_train_std=sc.fit_transform(X1_train)
X1_validation_std=sc.transform(X1_validation)
cov_mat1=np.cov(X1_validation_std.T)
pca1=PCA(n_components=12)
X1_train_pca=pca1.fit_transform(X1_train_std,Y1_train)
X1_validation_pca=pca1.fit_transform(X1_validation_std)
#Prog 2%
df2=df[["Rozpoznanie choroby:","Wiek","MORFOLOGIA 5 DIFF NEUTROFILE#", "ŻELAZO", "MOCZ - BADANIE OGÓLNE NABŁONKI PŁASKIE", "FERRYTYNA", "WITAMINA B12", "KREATYNINA W SUROWICY  EGFR", "25-OH-WITAMINA D", "MOCZ - BADANIE OGÓLNE UROBILINOGEN","MORFOLOGIA 5 DIFF MPV","KWAS FOLIOWY","CRP ULTRACZUŁE"]]
array = df2.values
X2=array[:,1:]
Y2=array[:,0]
validation_size = 0.20
seed = 7
X2_train, X2_validation, Y2_train, Y2_validation = model_selection.train_test_split(X2,Y2,test_size=validation_size, random_state=seed, shuffle=True)
X2_train_std=sc.fit_transform(X2_train)
X2_validation_std=sc.transform(X2_validation)
cov_mat2=np.cov(X2_validation_std.T)
pca2=PCA(n_components=12)
X2_train_pca=pca2.fit_transform(X2_train_std,Y2_train)
X2_validation_pca=pca2.fit_transform(X2_validation_std)
#Prog 5%
df5=df[["Rozpoznanie choroby:","Wiek","WITAMINA B12", "ŻELAZO","FERRYTYNA", "MOCZ - BADANIE OGÓLNE NABŁONKI PŁASKIE" ]]
array = df5.values
X5=array[:,1:]
Y5=array[:,0]
validation_size = 0.20
seed = 7
X5_train, X5_validation, Y5_train, Y5_validation = model_selection.train_test_split(X5,Y5,test_size=validation_size, random_state=seed, shuffle=True)
X5_train_std=sc.fit_transform(X5_train)
X5_validation_std=sc.transform(X5_validation)
cov_mat5=np.cov(X5_validation_std.T)
pca5=PCA(n_components=5)
X5_train_pca=pca5.fit_transform(X5_train_std,Y5_train)
X5_validation_pca=pca5.fit_transform(X5_validation_std)

#feat_labels=df2.columns[1:]
#forest=RandomForestClassifier(n_estimators=500,random_state=1)
#forest.fit(X2_train_pca,Y2_train)
#importances=forest.feature_importances_
#indices=np.argsort(importances)[::-1]
#for i in range(X2_train_pca.shape[1]):
#    print("%2d) %-*s %f" % (i+1,30, feat_labels[indices[i]], importances[indices[i]]))

#Test options and evaluation metric
seed = 7
scoring = 'accuracy'
#Spot Check Algorithms
models = []
#1
models.append((('Logistic Regression 1.1'), LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append((('Logistic Regression 1.2'), LogisticRegression(solver='liblinear', multi_class='auto')))
#models.append((('Logistic Regression 1.3'), LogisticRegression(solver='liblinear', multi_class='multinomial')))
#models.append((('Logistic Regression 1.4'), LogisticRegression(solver='liblinear', C=0.1, multi_class='ovr')))
#models.append((('Logistic Regression 1.5'), LogisticRegression(solver='liblinear', C=0.01, multi_class='ovr')))

#models.append((('Logistic Regression 2.1'), LogisticRegression(solver='liblinear', multi_class='ovr', class_weight='balanced')))
#models.append((('Logistic Regression 2.2'), LogisticRegression(solver='liblinear', multi_class='auto', class_weight='balanced')))
#models.append((('Logistic Regression 2.3'), LogisticRegression(solver='liblinear', multi_class='multinomial', class_weight='balanced')))
#models.append((('Logistic Regression 2.4'), LogisticRegression(solver='liblinear', C=0.1, multi_class='ovr', class_weight='balanced')))
#models.append((('Logistic Regression 2.5'), LogisticRegression(solver='liblinear', C=0.01, multi_class='ovr', class_weight='balanced')))

#models.append((('Logistic Regression 3.1'), LogisticRegression(solver='sag', multi_class='ovr', class_weight='balanced')))
#models.append((('Logistic Regression 3.2'), LogisticRegression(solver='sag', multi_class='auto', class_weight='balanced')))
#models.append((('Logistic Regression 3.3'), LogisticRegression(solver='sag', multi_class='multinomial', class_weight='balanced')))
#models.append((('Logistic Regression 3.4'), LogisticRegression(solver='sag', C=0.1, multi_class='ovr', class_weight='balanced')))
#models.append((('Logistic Regression 3.5'), LogisticRegression(solver='sag', C=0.01, multi_class='ovr', class_weight='balanced')))

#models.append((('Logistic Regression 4.1'), LogisticRegression(solver='saga', multi_class='ovr', class_weight='balanced')))
#models.append((('Logistic Regression 4.2'), LogisticRegression(solver='saga', multi_class='auto', class_weight='balanced')))
#models.append((('Logistic Regression 4.3'), LogisticRegression(solver='saga', multi_class='multinomial', class_weight='balanced')))
#models.append((('Logistic Regression 4.4'), LogisticRegression(solver='saga', C=0.1, multi_class='ovr', class_weight='balanced')))
#models.append((('Logistic Regression 4.5'), LogisticRegression(solver='saga', C=0.01, multi_class='ovr', class_weight='balanced')))


#models.append((('Logistic Regression 5.1'), LogisticRegression(penalty='l1', C=0.1, random_state=1,solver='sag', multi_class='ovr')))
#models.append((('Logistic Regression 5.2'), LogisticRegression(penalty='l1', C=0.01, random_state=1,solver='saga', multi_class='ovr')))
#models.append((('Logistic Regression 5.3'), LogisticRegression(penalty='l1', C=0.1, random_state=1,solver='sag', multi_class='auto')))
#models.append((('Logistic Regression 5.4'), LogisticRegression(penalty='l1', C=0.01, random_state=1,solver='saga', multi_class='auto')))
#models.append((('Logistic Regression 5.5'), LogisticRegression(penalty='l1', C=0.1, random_state=1,solver='sag', multi_class='multinomial')))
#models.append((('Logistic Regression 5.6'), LogisticRegression(penalty='l1', C=0.01, random_state=1,solver='saga', multi_class='multinomial')))

#2
models.append((('KNN'), KNeighborsClassifier()))
#models.append((('KNN 2'), KNeighborsClassifier(n_neighbors=3,p=1)))
#models.append((('KNN 3'), KNeighborsClassifier(n_neighbors=3,p=3)))
#models.append((('KNN 4'), KNeighborsClassifier(n_neighbors=5,p=1)))
#models.append((('KNN 5'), KNeighborsClassifier(n_neighbors=5,p=3)))
models.append((('KNN 6'), KNeighborsClassifier(n_neighbors=5,algorithm='kd_tree')))
models.append((('KNN 7'), KNeighborsClassifier(n_neighbors=5,algorithm='ball_tree')))
#models.append((('KNN 8'), KNeighborsClassifier(n_neighbors=5,p=1, weights='distance')))
#models.append((('KNN 9'), KNeighborsClassifier(n_neighbors=5,p=3, weights='distance')))
models.append((('KNN 10'), KNeighborsClassifier(n_neighbors=5, weights='distance')))
#models.append((('KNN 11'), KNeighborsClassifier(n_neighbors=5,p=0.1, weights='distance')))
#models.append((('KNN 12'), KNeighborsClassifier(n_neighbors=5,p=0.01, weights='distance')))
#models.append((('KNN 13'), KNeighborsClassifier(n_neighbors=5,p=1.5, weights='distance')))

#3
#models.append((('CART 1.1'), DecisionTreeClassifier()))
#models.append((('CART 1.2'), DecisionTreeClassifier(criterion='entropy')))
#models.append((('CART 1.3'), DecisionTreeClassifier(max_features=2)))
#models.append((('CART 1.4'), DecisionTreeClassifier(criterion='entropy', max_features=2)))
#models.append((('CART 1.5'), DecisionTreeClassifier(max_features=3)))
#models.append((('CART 1.6'), DecisionTreeClassifier(criterion='entropy', max_features=3)))
#models.append((('CART 1.7'), DecisionTreeClassifier(max_features=10)))
#models.append((('CART 1.8'), DecisionTreeClassifier(criterion='entropy', max_features=10)))
#models.append((('CART 1.9'), DecisionTreeClassifier(max_features='sqrt')))
#models.append((('CART 1.10'), DecisionTreeClassifier(criterion='entropy', max_features='sqrt')))
#models.append((('CART 1.11'), DecisionTreeClassifier(max_features='log2')))
#models.append((('CART 1.12'), DecisionTreeClassifier(criterion='entropy', max_features='log2')))
#models.append((('CART 1.13'), DecisionTreeClassifier(max_features='auto')))
#models.append((('CART 1.14'), DecisionTreeClassifier(criterion='entropy', max_features='auto')))

#models.append((('CART 2.1'), DecisionTreeClassifier(max_depth=5)))
#models.append((('CART 2.2'), DecisionTreeClassifier(max_depth=5, criterion='entropy')))
#models.append((('CART 2.3'), DecisionTreeClassifier(max_depth=5, max_features=2)))
#models.append((('CART 2.4'), DecisionTreeClassifier(max_depth=5, criterion='entropy', max_features=2)))
#models.append((('CART 2.5'), DecisionTreeClassifier(max_depth=5, max_features=3)))
#models.append((('CART 2.6'), DecisionTreeClassifier(max_depth=5, criterion='entropy', max_features=3)))
#models.append((('CART 2.7'), DecisionTreeClassifier(max_depth=5, max_features=10)))
#models.append((('CART 2.8'), DecisionTreeClassifier(max_depth=5, criterion='entropy', max_features=10)))
#models.append((('CART 2.9'), DecisionTreeClassifier(max_depth=5, max_features='sqrt')))
#models.append((('CART 2.10'), DecisionTreeClassifier(max_depth=5, criterion='entropy', max_features='sqrt')))
#models.append((('CART 2.11'), DecisionTreeClassifier(max_depth=5, max_features='log2')))
#models.append((('CART 2.12'), DecisionTreeClassifier(max_depth=5, criterion='entropy', max_features='log2')))
#models.append((('CART 2.13'), DecisionTreeClassifier(max_depth=5, max_features='auto')))
#models.append((('CART 2.14'), DecisionTreeClassifier(max_depth=5, criterion='entropy', max_features='auto')))

#4
models.append((('Random forest Bare 1.1'), RandomForestClassifier()))
#models.append((('Random forest Bare 1.2'),RandomForestClassifier(n_estimators=10)))
models.append((('Random forest Bare 1.3'),RandomForestClassifier( n_estimators=200)))
models.append((('Random forest Bare 1.4'),RandomForestClassifier(max_features='log2')))
models.append((('Random forest Bare 1.5'),RandomForestClassifier(max_features='sqrt')))
#models.append((('Random forest Bare 1.4.1'),RandomForestClassifier(max_features='log2', n_estimators=200)))
#models.append((('Random forest Bare 1.5.1'),RandomForestClassifier(max_features='sqrt', n_estimators=200)))
#models.append((('Random forest Bare 1.6'),RandomForestClassifier(n_estimators=10, criterion='entropy')))
models.append((('Random forest Bare 1.7'),RandomForestClassifier( n_estimators=200, criterion='entropy')))
#models.append((('Random forest Bare 1.7.1'),RandomForestClassifier( n_estimators=200, criterion='entropy',max_features='auto')))
#models.append((('Random forest Bare 1.7.2'),RandomForestClassifier( n_estimators=200, criterion='entropy',max_features='log2')))
models.append((('Random forest Bare 1.8'),RandomForestClassifier(max_features='auto', criterion='entropy')))
models.append((('Random forest Bare 1.9'),RandomForestClassifier(max_features='log2', criterion='entropy')))
#models.append((('Random forest Bare 1.10'),RandomForestClassifier(max_features='sqrt', criterion='entropy')))
#models.append((('Random forest Bare 1.11'), RandomForestClassifier(bootstrap=False)))
#models.append((('Random forest Bare 1.12'),RandomForestClassifier(n_estimators=10,bootstrap=False)))
#models.append((('Random forest Bare 1.13'),RandomForestClassifier( n_estimators=200,bootstrap=False)))
#models.append((('Random forest Bare 1.14'),RandomForestClassifier(max_features='log2',bootstrap=False)))
#models.append((('Random forest Bare 1.15'),RandomForestClassifier(max_features='sqrt',bootstrap=False)))
#models.append((('Random forest Bare 1.16'),RandomForestClassifier(n_estimators=10, criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 1.17'),RandomForestClassifier( n_estimators=200, criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 1.18'),RandomForestClassifier(max_features='auto', criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 1.19'),RandomForestClassifier(max_features='log2', criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 1.20'),RandomForestClassifier(max_features='sqrt', criterion='entropy',bootstrap=False)))

models.append((('Random forest Bare 2.1'), RandomForestClassifier(max_depth=5)))
#models.append((('Random forest Bare 2.2'),RandomForestClassifier(max_depth=5,n_estimators=10)))
#models.append((('Random forest Bare 2.3'),RandomForestClassifier(max_depth=5, n_estimators=200)))
#models.append((('Random forest Bare 2.4'),RandomForestClassifier(max_depth=5,max_features='log2')))
#models.append((('Random forest Bare 2.5'),RandomForestClassifier(max_depth=5,max_features='sqrt')))
#models.append((('Random forest Bare 2.6'),RandomForestClassifier(max_depth=5,n_estimators=10, criterion='entropy')))
#models.append((('Random forest Bare 2.7'),RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy')))
#models.append((('Random forest Bare 2.8'),RandomForestClassifier(max_depth=5,max_features='auto', criterion='entropy')))
#models.append((('Random forest Bare 2.9'),RandomForestClassifier(max_depth=5,max_features='log2', criterion='entropy')))
models.append((('Random forest Bare 2.10'),RandomForestClassifier(max_depth=5,max_features='sqrt', criterion='entropy')))
#models.append((('Random forest Bare 2.11'), RandomForestClassifier(max_depth=5,bootstrap=False)))
#models.append((('Random forest Bare 2.12'),RandomForestClassifier(max_depth=5,n_estimators=10,bootstrap=False)))
#models.append((('Random forest Bare 2.13'),RandomForestClassifier(max_depth=5, n_estimators=200,bootstrap=False)))
#models.append((('Random forest Bare 2.14'),RandomForestClassifier(max_depth=5,max_features='log2',bootstrap=False)))
#models.append((('Random forest Bare 2.15'),RandomForestClassifier(max_depth=5,max_features='sqrt',bootstrap=False)))
#models.append((('Random forest Bare 2.16'),RandomForestClassifier(max_depth=5,n_estimators=10, criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 2.17'),RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 2.18'),RandomForestClassifier(max_depth=5,max_features='auto', criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 2.19'),RandomForestClassifier(max_depth=5,max_features='log2', criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 2.20'),RandomForestClassifier(max_depth=5,max_features='sqrt', criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 2.21'),RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy',max_features='log2')))
#models.append((('Random forest Bare 2.22'),RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy',max_features='auto')))
#models.append((('Random forest Bare 2.23'),RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy',max_features='log2',bootstrap=False)))
#models.append((('Random forest Bare 2.24'),RandomForestClassifier(max_depth=5, n_estimators=200, criterion='entropy',max_features='auto',bootstrap=False)))

#models.append((('Random forest Bare 3.1'), RandomForestClassifier(max_depth=3)))
#models.append((('Random forest Bare 3.2'),RandomForestClassifier(max_depth=3,n_estimators=10)))
#models.append((('Random forest Bare 3.3'),RandomForestClassifier(max_depth=3, n_estimators=200)))
#models.append((('Random forest Bare 3.4'),RandomForestClassifier(max_depth=3,max_features='log2')))
#models.append((('Random forest Bare 3.5'),RandomForestClassifier(max_depth=3,max_features='sqrt')))
#models.append((('Random forest Bare 3.6'),RandomForestClassifier(max_depth=3,n_estimators=10, criterion='entropy')))
#models.append((('Random forest Bare 3.7'),RandomForestClassifier(max_depth=3, n_estimators=200, criterion='entropy')))
#models.append((('Random forest Bare 3.8'),RandomForestClassifier(max_depth=3,max_features='auto', criterion='entropy')))
#models.append((('Random forest Bare 3.9'),RandomForestClassifier(max_depth=3,max_features='log2', criterion='entropy')))
#models.append((('Random forest Bare 3.10'),RandomForestClassifier(max_depth=3,max_features='sqrt', criterion='entropy')))
#models.append((('Random forest Bare 3.11'), RandomForestClassifier(max_depth=3,bootstrap=False)))
#models.append((('Random forest Bare 3.12'),RandomForestClassifier(max_depth=3,n_estimators=10,bootstrap=False)))
#models.append((('Random forest Bare 3.13'),RandomForestClassifier(max_depth=3, n_estimators=200,bootstrap=False)))
#models.append((('Random forest Bare 3.14'),RandomForestClassifier(max_depth=3,max_features='log2',bootstrap=False)))
#models.append((('Random forest Bare 3.15'),RandomForestClassifier(max_depth=3,max_features='sqrt',bootstrap=False)))
#models.append((('Random forest Bare 3.16'),RandomForestClassifier(max_depth=3,n_estimators=10, criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 3.17'),RandomForestClassifier(max_depth=3, n_estimators=200, criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 3.18'),RandomForestClassifier(max_depth=3,max_features='auto', criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 3.19'),RandomForestClassifier(max_depth=3,max_features='log2', criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 3.20'),RandomForestClassifier(max_depth=3,max_features='sqrt', criterion='entropy',bootstrap=False)))
#models.append((('Random forest Bare 2.21'),RandomForestClassifier(max_depth=3, n_estimators=200, criterion='entropy',max_features='log2',bootstrap=False)))
#models.append((('Random forest Bare 2.22'),RandomForestClassifier(max_depth=3, n_estimators=200, criterion='entropy',max_features='auto',bootstrap=False)))



#models.append((('Gaussian Naive Bayes'), GaussianNB()))
#models.append((('Perceptron eta0=1'), Perceptron(eta0=0.1, random_state=1)))
#models.append((('Perceptron bare'), Perceptron()))
#models.append((('MultiLayer Perceptron'), MLPClassifier(random_state=1, max_iter=300)))
#models.append((('QDA'), QuadraticDiscriminantAnalysis()))
#5 
#models.append((('GBC 1.1'), GradientBoostingClassifier()))
models.append((('GBC 1.2'), GradientBoostingClassifier(loss='exponential')))
#models.append((('GBC 1.3'), GradientBoostingClassifier(n_estimators=10)))
#models.append((('GBC 1.4'), GradientBoostingClassifier(n_estimators=10,loss='exponential')))
models.append((('GBC 1.5'), GradientBoostingClassifier(n_estimators=200)))
models.append((('GBC 1.6'), GradientBoostingClassifier(n_estimators=200,loss='exponential')))
#models.append((('GBC 1.7'), GradientBoostingClassifier(learning_rate=0.01)))
#models.append((('GBC 1.8'), GradientBoostingClassifier(learning_rate=0.01,loss='exponential')))
#models.append((('GBC 1.9'), GradientBoostingClassifier(learning_rate=0.01,n_estimators=10)))
#models.append((('GBC 1.10'), GradientBoostingClassifier(learning_rate=0.01,n_estimators=10,loss='exponential')))
#models.append((('GBC 1.11'), GradientBoostingClassifier(learning_rate=0.01,n_estimators=200)))
#models.append((('GBC 1.12'), GradientBoostingClassifier(learning_rate=0.01,n_estimators=200,loss='exponential')))
#models.append((('GBC 1.13'), GradientBoostingClassifier(learning_rate=1.0)))
#models.append((('GBC 1.14'), GradientBoostingClassifier(learning_rate=1.0,loss='exponential')))
#models.append((('GBC 1.15'), GradientBoostingClassifier(learning_rate=1.0,n_estimators=10)))
#models.append((('GBC 1.16'), GradientBoostingClassifier(learning_rate=1.0,n_estimators=10,loss='exponential')))
models.append((('GBC 1.17'), GradientBoostingClassifier(learning_rate=1.0,n_estimators=200)))
#models.append((('GBC 1.18'), GradientBoostingClassifier(learning_rate=1.0,n_estimators=200,loss='exponential')))

models.append((('GBC 2.1'), GradientBoostingClassifier(criterion='mse')))
models.append((('GBC 2.2'), GradientBoostingClassifier(criterion='mse',loss='exponential')))
#models.append((('GBC 2.3'), GradientBoostingClassifier(criterion='mse',n_estimators=10)))
#models.append((('GBC 2.4'), GradientBoostingClassifier(criterion='mse',n_estimators=10,loss='exponential')))
models.append((('GBC 2.5'), GradientBoostingClassifier(criterion='mse',n_estimators=200)))
models.append((('GBC 2.6'), GradientBoostingClassifier(criterion='mse',n_estimators=200,loss='exponential')))
#models.append((('GBC 2.7'), GradientBoostingClassifier(criterion='mse',learning_rate=0.01)))
#models.append((('GBC 2.8'), GradientBoostingClassifier(criterion='mse',learning_rate=0.01,loss='exponential')))
#models.append((('GBC 2.9'), GradientBoostingClassifier(criterion='mse',learning_rate=0.01,n_estimators=10)))
#models.append((('GBC 2.10'), GradientBoostingClassifier(criterion='mse',learning_rate=0.01,n_estimators=10,loss='exponential')))
#models.append((('GBC 2.11'), GradientBoostingClassifier(criterion='mse',learning_rate=0.01,n_estimators=200)))
#models.append((('GBC 2.12'), GradientBoostingClassifier(criterion='mse',learning_rate=0.01,n_estimators=200,loss='exponential')))
#models.append((('GBC 2.13'), GradientBoostingClassifier(criterion='mse',learning_rate=1.0)))
#models.append((('GBC 2.14'), GradientBoostingClassifier(criterion='mse',learning_rate=1.0,loss='exponential')))
#models.append((('GBC 2.15'), GradientBoostingClassifier(criterion='mse',learning_rate=1.0,n_estimators=10)))
#models.append((('GBC 2.16'), GradientBoostingClassifier(criterion='mse',learning_rate=1.0,n_estimators=10,loss='exponential')))
#models.append((('GBC 2.17'), GradientBoostingClassifier(criterion='mse',learning_rate=1.0,n_estimators=200)))
#models.append((('GBC 2.18'), GradientBoostingClassifier(criterion='mse',learning_rate=1.0,n_estimators=200,loss='exponential')))

#models.append((('GBC 3.1'), GradientBoostingClassifier(criterion='mae')))
#models.append((('GBC 3.2'), GradientBoostingClassifier(criterion='mae',loss='exponential')))
#models.append((('GBC 3.3'), GradientBoostingClassifier(criterion='mae',n_estimators=10)))
#models.append((('GBC 3.4'), GradientBoostingClassifier(criterion='mae',n_estimators=10,loss='exponential')))
#models.append((('GBC 3.5'), GradientBoostingClassifier(criterion='mae',n_estimators=200)))
#models.append((('GBC 3.6'), GradientBoostingClassifier(criterion='mae',n_estimators=200,loss='exponential')))
#models.append((('GBC 3.7'), GradientBoostingClassifier(criterion='mae',learning_rate=0.01)))
#models.append((('GBC 3.8'), GradientBoostingClassifier(criterion='mae',learning_rate=0.01,loss='exponential')))
#models.append((('GBC 3.9'), GradientBoostingClassifier(criterion='mae',learning_rate=0.01,n_estimators=10)))
#models.append((('GBC 3.10'), GradientBoostingClassifier(criterion='mae',learning_rate=0.01,n_estimators=10,loss='exponential')))
#models.append((('GBC 3.11'), GradientBoostingClassifier(criterion='mae',learning_rate=0.01,n_estimators=200)))
#models.append((('GBC 3.12'), GradientBoostingClassifier(criterion='mae',learning_rate=0.01,n_estimators=200,loss='exponential')))
#models.append((('GBC 3.13'), GradientBoostingClassifier(criterion='mae',learning_rate=1.0)))
#models.append((('GBC 3.14'), GradientBoostingClassifier(criterion='mae',learning_rate=1.0,loss='exponential')))
#models.append((('GBC 3.15'), GradientBoostingClassifier(criterion='mae',learning_rate=1.0,n_estimators=10)))
#models.append((('GBC 3.16'), GradientBoostingClassifier(criterion='mae',learning_rate=1.0,n_estimators=10,loss='exponential')))
#models.append((('GBC 3.17'), GradientBoostingClassifier(criterion='mae',learning_rate=1.0,n_estimators=200)))
#models.append((('GBC 3.18'), GradientBoostingClassifier(criterion='mae',learning_rate=1.0,n_estimators=200,loss='exponential')))


#models.append((('Bagging'), BaggingClassifier()))
#models.append((('AdaBoost'), AdaBoostClassifier()))
#estimators = [('Random Forest est=10', RandomForestClassifier(n_estimators=10, random_state=42)), ('svr', make_pipeline(StandardScaler(),LinearSVC(random_state=42)))]
#models.append((('GBM'), HistGradientBoostingClassifier()))
#models.append((('GBRT'), GradientBoostingClassifier()))
#models.append((('GBR'),  GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, random_state=0, loss='ls')))
#models.append((('SVM 1.1'), SVC()))
models.append((('SVM 1.2'), SVC(probability=True)))
#models.append((('SVM 1.3'), SVC(probability=True, C=0.025)))
#models.append((('SVM 1.4'), SVC(C=0.025)))
models.append((('SVM 1.5'), SVC(gamma='auto',probability=True)))
#models.append((('SVM 1.6'), SVC(gamma='auto',probability=True, C=0.025)))
models.append((('SVM 1.7'), SVC(probability=True, C=2.5)))
#models.append((('SVM 1.8'), SVC(C=2.5)))
models.append((('SVM 1.9'), SVC(gamma='auto',probability=True, C=2.5)))

#models.append((('SVM 2.1'), SVC(kernel="linear")))
models.append((('SVM 2.2'), SVC(kernel="linear",probability=True)))
models.append((('SVM 2.3'), SVC(kernel="linear",probability=True, C=0.025)))
#models.append((('SVM 2.4'), SVC(kernel="linear",C=0.025)))
models.append((('SVM 2.5'), SVC(kernel="linear",gamma='auto',probability=True)))
models.append((('SVM 2.6'), SVC(kernel="linear",gamma='auto',probability=True, C=0.025)))
models.append((('SVM 2.7'), SVC(kernel="linear",probability=True, C= 2.5)))
#models.append((('SVM 2.8'), SVC(kernel="linear",C=2.5)))
models.append((('SVM 2.9'), SVC(kernel="linear",gamma='auto',probability=True, C=2.5)))

#models.append((('SVM 3.1'), SVC(kernel="poly")))
#models.append((('SVM 3.2'), SVC(kernel="poly",probability=True)))
#models.append((('SVM 3.3'), SVC(kernel="poly",probability=True, C=0.025)))
#models.append((('SVM 3.4'), SVC(kernel="poly",C=0.025)))
#models.append((('SVM 3.5'), SVC(kernel="poly",gamma='auto',probability=True)))
#models.append((('SVM 3.6'), SVC(kernel="poly",gamma='auto',probability=True, C=0.025)))
#models.append((('SVM 3.7'), SVC(kernel="poly",probability=True, C=2.5)))
#models.append((('SVM 3.8'), SVC(kernel="poly",C=2.5)))
#models.append((('SVM 3.9'), SVC(kernel="poly",gamma='auto',probability=True, C=2.5)))
#models.append((('SVM 3.10'), SVC(kernel="poly", degree=1)))
#models.append((('SVM 3.11'), SVC(kernel="poly", degree=3)))
#models.append((('SVM 3.12'), SVC(kernel="poly", coef0=1)))
#models.append((('SVM 3.13'), SVC(kernel="poly",probability=True, degree=3)))
#models.append((('SVM 3.14'), SVC(kernel="poly",probability=True, C=0.025, degree=3)))
#models.append((('SVM 3.15'), SVC(kernel="poly",C=0.025, degree=3)))
#models.append((('SVM 3.16'), SVC(kernel="poly",gamma='auto',probability=True, degree=3)))
#models.append((('SVM 3.17'), SVC(kernel="poly",gamma='auto',probability=True, C=0.025, degree=3)))
#models.append((('SVM 3.18'), SVC(kernel="poly",probability=True, C=2.5, degree=3)))
#models.append((('SVM 3.19'), SVC(kernel="poly",C=2.5, degree=3)))
#models.append((('SVM 3.20'), SVC(kernel="poly",gamma='auto',probability=True, C=2.5, degree=3)))
#models.append((('SVM 3.21'), SVC(kernel="poly",probability=True, degree=1)))
#models.append((('SVM 3.22'), SVC(kernel="poly",probability=True, C=0.025, degree=1)))
#models.append((('SVM 3.23'), SVC(kernel="poly",C=0.025, degree=1)))
models.append((('SVM 3.24'), SVC(kernel="poly",gamma='auto',probability=True, degree=1)))
models.append((('SVM 3.25'), SVC(kernel="poly",gamma='auto',probability=True, C=0.025, degree=1)))
models.append((('SVM 3.26'), SVC(kernel="poly",probability=True, C=2.5, degree=1)))
#models.append((('SVM 3.27'), SVC(kernel="poly",C=2.5, degree=1)))
models.append((('SVM 3.28'), SVC(kernel="poly",gamma='auto',probability=True, C=2.5, degree=1)))
models.append((('SVM 3.29'), SVC(kernel="poly",probability=True, coef0=1)))
models.append((('SVM 3.30'), SVC(kernel="poly",probability=True, C=0.025, coef0=1)))
#models.append((('SVM 3.31'), SVC(kernel="poly",C=0.025, coef0=1)))
#models.append((('SVM 3.32'), SVC(kernel="poly",gamma='auto',probability=True, coef0=1)))
models.append((('SVM 3.33'), SVC(kernel="poly",gamma='auto',probability=True, C=0.025, coef0=1)))
models.append((('SVM 3.34'), SVC(kernel="poly",probability=True, C=2.5, coef0=1)))
#models.append((('SVM 3.35'), SVC(kernel="poly",C=2.5, coef0=1)))
#models.append((('SVM 3.36'), SVC(kernel="poly",gamma='auto',probability=True, C=2.5, coef0=1)))

#models.append((('SVM 4.1'), SVC(kernel="precomputed")))
#models.append((('SVM 4.2'), SVC(kernel="precomputed",probability=True)))
#models.append((('SVM 4.3'), SVC(kernel="precomputed",probability=True, C=0.025)))
#models.append((('SVM 4.4'), SVC(kernel="precomputed",C=0.025)))
#models.append((('SVM 4.5'), SVC(kernel="precomputed",gamma='auto',probability=True)))
#models.append((('SVM 4.6'), SVC(kernel="precomputed",gamma='auto',probability=True, C=0.025)))
#models.append((('SVM 4.7'), SVC(kernel="precomputed",probability=True, C=2.5)))
#models.append((('SVM 4.8'), SVC(kernel="precomputed",C=2.5)))
#models.append((('SVM 4.9'), SVC(kernel="precomputed",gamma='auto',probability=True, C=2.5)))

#models.append((('SVM 5.1'), SVC(kernel="poly")))
#models.append((('SVM 5.2'), SVC(kernel="poly",probability=True)))
#models.append((('SVM 5.3'), SVC(kernel="poly",probability=True, C=0.025)))
#models.append((('SVM 5.4'), SVC(kernel="poly",C=0.025)))
#models.append((('SVM 5.5'), SVC(kernel="poly",gamma='auto',probability=True)))
#models.append((('SVM 5.6'), SVC(kernel="poly",gamma='auto',probability=True, C=0.025)))
#models.append((('SVM 5.7'), SVC(kernel="poly",probability=True, C=2.5)))
#models.append((('SVM 5.8'), SVC(kernel="poly",C=2.5)))
#models.append((('SVM 5.9'), SVC(kernel="poly",gamma='auto',probability=True, C=2.5)))
models.append((('SVM 5.10'), SVC(kernel="poly",probability=True, coef0=1)))
models.append((('SVM 5.11'), SVC(kernel="poly",probability=True, C=0.025, coef0=1)))
#models.append((('SVM 5.12'), SVC(kernel="poly",C=0.025, coef0=1)))
models.append((('SVM 5.13'), SVC(kernel="poly",gamma='auto',probability=True, coef0=1)))
models.append((('SVM 5.14'), SVC(kernel="poly",gamma='auto',probability=True, C=0.025, coef0=1)))
models.append((('SVM 5.15'), SVC(kernel="poly",probability=True, C=2.5, coef0=1)))
#models.append((('SVM 5.16'), SVC(kernel="poly",C=2.5, coef0=1)))
#models.append((('SVM 5.17'), SVC(kernel="poly",gamma='auto',probability=True, C=2.5, coef0=1)))
#Grupowe
#models.append((('MVC'),MajorityVoteClassifier(classifiers=models)))
#models.append((('MVC'),BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', random_state=1), n_estimators=500,n_jobs=1,random_state=1)))

#Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train_pca, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
vc=VotingClassifier(estimators=models,voting='soft')
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
cv_results = model_selection.cross_val_score(vc, X_train_pca, Y_train, cv=kfold, scoring=scoring)
results.append(cv_results)
name='Voting Classifier'
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
print("------------------------------------------------------------")
#Prog 1%
results = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X1_train_pca, Y1_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
vc=VotingClassifier(estimators=models,voting='soft')
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
cv_results = model_selection.cross_val_score(vc, X1_train_pca, Y1_train, cv=kfold, scoring=scoring)
results.append(cv_results)
name='Voting Classifier'
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
print("------------------------------------------------------------")
#Prog 2%
results = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X2_train_pca, Y2_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
vc=VotingClassifier(estimators=models,voting='soft')
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
cv_results = model_selection.cross_val_score(vc, X2_train_pca, Y2_train, cv=kfold, scoring=scoring)
results.append(cv_results)
name='Voting Classifier'
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
print("------------------------------------------------------------")
#Prog 5%
results = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X5_train_pca, Y5_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
vc=VotingClassifier(estimators=models,voting='soft')
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
cv_results = model_selection.cross_val_score(vc, X5_train_pca, Y5_train, cv=kfold, scoring=scoring)
results.append(cv_results)
name='Voting Classifier'
names.append(name)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)
print("------------------------------------------------------------")


#Istotnosc cech
#feat_labels=df.columns[1:]
#forest=RandomForestClassifier(n_estimators=500,random_state=1)
#forest.fit(X_train_pca,Y_train_pca)
#importances=forest.feature_importances_
#indices=np.argsort(importances)[::-1]
#for i in range(X_train_pca.shape[1]):
#    print("%2d) %-*s %f" % (i+1,30, feat_labels[indices[i]], importances[indices[i]]))
