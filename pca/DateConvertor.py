import numpy as np
import pandas as pd
from datetime import datetime
import time;
import glob
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

file_Data = pd.read_csv('lcms_outgoing_sms_offnet_count123.csv',sep= ',', header= None)
Date =file_Data.values[:,0]


MSISDN = file_Data.values[:,1]
Count=  file_Data.values[:,2]

print("Preprocessing is Started")
d = []
for datee in range(len(Date)):    
    res = datetime.strptime(str(Date[datee]), "%Y-%m-%d %H:%M:%S.000+0000")
    
    stringdate = str(res)
    m = stringdate[0:10]   
    t = time.mktime(time.strptime(m, "%Y-%m-%d"));    
    d.append(t)

raw_data = {'Date': d,
            'MSISDN': MSISDN,
            'Count': Count}
df = pd.DataFrame(raw_data, columns = ['Date', 'MSISDN','Count'])


df.to_csv('lcms_outgoing_sms_offnet_count1.csv',index = False)
myfiles = glob.glob("lcms_outgoing_sms_offnet_count1.csv")
for file in myfiles:
    lines = open(file).readlines()
    open(file, 'w').writelines(lines[1:])
print("--------------Preprocessing is done---------------")

print("Applying PCA")

new_File_Data = pd.read_csv('lcms_outgoing_sms_offnet_count1.csv',sep= ',', header= None)
X = new_File_Data.values[:, 0:3]
pca = PCA(n_components=3)
sklearn_transf = pca.fit_transform(X)


plt.plot(sklearn_transf[0:2,0],sklearn_transf[0:2,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
##plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

plt.show()


##sklearn_transf  = pca.fit(X)
##first = pca.components_[0]
##second = pca.components_[1]
##third = pca.components_[2]

##transformed_data = pca.transform(X)
##for ii,jj in zip(transformed_data,X):
##    plt.scatter(first[0]*ii[0],first[1]*ii[0],color ='r')
##    plt.scatter(second[0]*jj[1],second[1]*jj[1],color ='b')
##    ##plt.scatter(third[0]*kk[0],third[1]*kk[0],color ='c')
##    plt.scatter(jj[0],jj[1],color ='c')
##plt.show()

##var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
##plt.plot(var1)
##plt.show()

##plt.scatter(first,second,third)
##plt.show()



print(pca.explained_variance_ratio_)

print("Applying PCA is done!!!!!!!!!!!!!")
