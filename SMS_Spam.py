import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import plotly_express as px
import wordcloud
data=pd.read_csv('SPAMcsv.csv')

# Clear and organize Data
data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
data.rename(columns = {'v1':'class_label','v2':'TaxtMessage'},inplace=True)
data['length'] = data['TaxtMessage'].apply(len)
print(data.columns)

# distribution of labels:

#fig=px.histogram(data,x='class_label',color="class_label", color_discrete_sequence=["#F71A00","#1A1981"])
#fig = px.pie(data.class_label.value_counts(),labels='index', values='class_label', color="class_label", color_discrete_sequence=["#871fff","#ffa78c"] )
#fig.show()


#Model

X=data.iloc[:,1]
y=data.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=20)


Count_Vec=CountVectorizer();
features=Count_Vec.fit_transform(x_train)
print(features)


#SVM Model
Model=SVC()
Model.fit(features,y_train)
X_test=Count_Vec.transform(x_test)
Score=Model.score(X_test,y_test)
print('Accuracy = {:.2%}'.format(Score))

