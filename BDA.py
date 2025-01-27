
Program 1

mapper.py
#!/usr/bin/env python3 
import sys
for line in sys.stdin: 
# Remove whitespaces and split the line
line = line.strip() 
words = line.split() 
# Output each word's individual count
for word in words: 
print(word, '\t', 1)
reducer.py
#!/usr/bin/env python3
from collections import defaultdict
import sys
word_counts = defaultdict(int) # Stores words and counts
for line insys.stdin:
try:
# Remove whitespaces and split the line
line = line.strip()
word, count = line.split()
# Get each word's individual count
count = int(count)
except:
continue
# Add the count word-wise
word_counts[word] += count
# Output word counts
for word, count in word_counts.items():
print(word, count)




Program 2

mapper.py
#!/usr/bin/env python
import re
import sys
for line in sys.stdin:
val = line.strip()
(year, temp, q) = (val[15:19], val[87:92], val[92:93])
if (temp != "+9999" and re.match("[01459]", q)):
print("%s\t%s" % (year, temp))
reducer.py
#!/usr/bin/env python
import sys
(last_key, max_val) = (None, 0)
for line in sys.stdin:
(key, val) = line.strip().split("\t")
if last_key and last_key != key:
print("%s\t%s" % (last_key, max_val))
(last_key, max_val) = (key, int(val))
else:
(last_key, max_val) = (key, max(max_val, int(val)))
if last_key:
print("%s\t%s" % (last_key, max_val))




Program 3

mapper.py 
#!/usr/bin/env python 
import sys 
for line in sys.stdin: 
# Remove leading and trailing whitespace 
line = line.strip() 
# Split the line into columns 
columns = line.split('\t') 
# Extract the salary column 
salary = columns[2] 
# Output the salary with a null key 
print("%s\t%s" % (None, salary)) 
reducer.py 
import sys 
# Initialize the maximum salary to 0 
max_salary = 0 
# Iterate over each line in the input 
for line in sys.stdin: 
# Split the line into key and value 
key, value = line.strip().split('\t', 1) 
# Convert the value to an integer 
try: 
salary = int(value) 
except ValueError: 
continue 
# Update the maximum salary if this salary is higher 
if salary > max_salary: 
max_salary = salary 
# Output the final maximum salary 
print("Max Salary: %s" % max_salary)



Program 4


mapper.py 
#!/usr/bin/env python 
import sys 
import csv 
for line in csv.reader(iter(sys.stdin.readline, '')): 
year = line[0] 
sales = line[2] 
print(f"{year}\t{sales}") 
reducer.py 
#!/usr/bin/env python 
import sys 
current_year = None 
total_sales = 0 
for line in sys.stdin: 
year, sales = line.strip().split('\t') 
if current_year is None: 
current_year = year 
if year == current_year: 
total_sales += int(sales) 
else: 
print(f"{current_year}\t{total_sales}") 
current_year = year 
total_sales = int(sales) 
if current_year is not None: 
print(f"{current_year}\t{total_sales}")



Program 5


mapper.py 
#!/usr/bin/envpython
import string
import fileinput
for line in fileinput.input():
data = line.strip().split("\t")
if len(data) == 6:
date, time, location, item, cost, payment = data
print "{0}\t{1}".format(location, cost)
reducer.py
#!/usr/bin/env python
import fileinput
max_value = 0
old_key = None
for line in fileinput.input():
data = line.strip().split("\t")    
if len(data) != 2:
# Something has gone wrong. Skip this line.
continue
current_key, current_value = data
# Refresh for new keys (i.e. locations in the example context)
if old_key and old_key != current_key:
print old_key, "\t", max_value
old_key = current_key
max_value = 0
old_key = current_key
if float(current_value) > float(max_value):
max_value = float(current_value)
if old_key != None:
print old_key, "\t", max_value










Program 6

Linear Regression 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
dataset = pd.read_csv('salary_Data.csv') 
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 1].values 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression() 
regressor.fit(X_train, y_train) 
y_pred = regressor.predict(X_test) 
#Visualizing Training set results 
plt.scatter(X_train, y_train, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue') 
plt.title('Salary vs Experience {Training set}') 
plt.xlabel('Years of experience') 
plt.ylabel('Salary') 
plt.show() 
#Visualizing Test set results 
plt.scatter(X_test, y_test, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue') 
plt.title('Salary vs Experience {Test set}') 
plt.xlabel('Years of experience') 
plt.ylabel('Salary') 
plt.show()


Program 7

from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn import svm 
from sklearn import metrics 
cancer_data = datasets.load_breast_cancer() 
print(cancer_data.data[5]) 
print(cancer_data.data.shape) 
#target set 
print(cancer_data.target) 
cancer_data = datasets.load_breast_cancer() 
X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.4,random_state=109) 
#create a classifier 
cls = svm.SVC(kernel="linear") 
#train the model 
cls.fit(X_train,y_train) 
#predict the response 
pred = cls.predict(X_test) 
#accuracy 
print("acuracy:", metrics.accuracy_score(y_test,y_pred=pred)) 
#precision score 
print("precision:", metrics.precision_score(y_test,y_pred=pred)) 
#recall score 
print("recall" , metrics.recall_score(y_test,y_pred=pred)) 
print(metrics.classification_report(y_test, y_pred=pred))



Program 8

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.cluster import DBSCAN 
data = pd.read_csv("Mall_Customers.csv") 
data.head() 
print("Dataset shape:", data.shape) 
data.isnull().any().any() 
x = data.loc[:, ['Annual Income (k$)','Spending Score (1-100)']].values 
# cluster the data into five clusters 
dbscan = DBSCAN(eps = 8, min_samples = 4).fit(x) 
# fitting the model 
labels = dbscan.labels_ # getting the labels 
plt.scatter(x[:, 0], x[:,1], c = labels, cmap= "plasma") 
# plotting the clusters 
plt.xlabel("Income") # X-axis label 
plt.ylabel("Spending Score") # Y-axis label 
plt.show() # showing the plot


Program 9

from sklearn.datasets import make_blobs 
from sklearn.cluster import OPTICS 
import numpy as np 
import matplotlib.pyplot as plt 
# Configuration options 
num_samples_total = 1000 
cluster_centers = [(3,3), (7,7)] 
num_classes = len(cluster_centers) 
epsilon = 2.0 
min_samples = 22 
cluster_method = 'xi' 
metric = 'minkowski' 
# Generate data 
X, y = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_classes, center_box=(0, 1), cluster_std = 0.5) 
# Compute OPTICS 
db = OPTICS(max_eps=epsilon, min_samples=min_samples, cluster_method=cluster_method, metric=metric).fit(X) 
labels = db.labels_ 
no_clusters = len(np.unique(labels) ) 
no_noise = np.sum(np.array(labels) == -1, axis=0) 
print('Estimated no. of clusters: %d' % no_clusters) 
print('Estimated no. of noise points: %d' % no_noise) 
# Generate scatter plot for training data 
colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels)) 
plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True) 
plt.title(f'OPTICS clustering') 
plt.xlabel('Axis X[0]') 
plt.ylabel('Axis X[1]')
plt.show()



Program 10

i) Plot bar chart for number of parts in each category of toys 
import pandas as pd 
import matplotlib.pyplot as plt 
df=pd.read_csv('parts.csv') 
sd =df.groupby('part_cat_id') ['part_cat_id'].count().sort_values(ascending=False) 
y=sd.values 
x=sd.index 
plt.title('Bar Chart') 
plt.xlabel('Part Category id') 
plt.ylabel('Category Count') 
plt.bar(x, y, color ='gray') 
plt.show()



ii)Display the Scatter Plot 
import matplotlib.pyplot as plt 
import pandas as pd
df= pd.read_csv('sets.csv') 
X = df['theme_id'].values 
Y= df['num_parts'].values 
plt.scatter( X, Y, color='blue',s=10) 
plt.xlabel('Theme Id') 
plt.ylabel('Number of parts') 
# as(x_min, x_max, y_min , y_max) 
X_min =0 
X_max=40 
Y_min=-1 
Y_max=50 
plt.axis([X_min, X_max, Y_min, Y_max]) 
plt.title('Scatter plot') 
plt.show()


iii) Pie chart 
import pandas as pd 
import matplotlib.pyplot as plt 
df = pd.read_csv('colors.csv')

sd=df.groupby(['is_trans'])['is_trans'].count() 
y=sd.values #Extracts count values in array a 
expl =(0.1,0) 
labels = 'Non-Transparent', 'Transparent' 
colors=['red','grey'] 
plt.pie(y, labels=labels, explode=expl, colors=colors, autopct='%1.1f%%',shadow=True, startangle=90) 
plt.axis('equal') 
plt.show()

