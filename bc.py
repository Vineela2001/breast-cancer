import pandas as pd
import pickle

ds=pd.read_csv('breast-cancer-wisconsin.data')
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
ds.iloc[:,6]=le.fit_transform(ds.iloc[:,6].values)
X=ds.iloc[:,1:-1].values
Y=ds.iloc[:,10].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25)
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train,y_train)
predictions=reg.predict(x_test)
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test,predictions)
print(cm)
sns.heatmap(cm,annot=True)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
import pickle
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(reg, f)

#De-Serializing the model
with open('trained_model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)
    

#Check the pickle file by inputing the variables
model = pickle.load(open('trained_model.pkl','rb'))
print(model.predict([[4,8,8,5,4,2,1,1,1]]))
