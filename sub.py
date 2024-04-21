from tkinter import _test
from xml.sax.handler import all_features
import pandas as pd
autos=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/autos.csv')
comedy=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/comedy.csv')
education=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/education.csv')
entertainment=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/entertainment.csv')
film=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/film.csv')
gaming=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/gaming.csv')
how=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/how.csv')
music=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/music.csv')
news=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/news.csv')
non_profits=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/nonprofits.csv')
pets=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/pets.csv')
people=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/people.csv')
science=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/science.csv')
sports=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/sports.csv')
travel=pd.read_csv('D:/Users/Dell/Documents/aml/subdata/travel.csv')


from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

test=pd.read_csv('D:/Users/Dell/Documents/aml/clean.csv')

def evaluate(train_title,train_desc,train_y,test_title,test_desc):
    cv = CountVectorizer(max_features=5000)
    le = LabelEncoder()
    X_train=cv.fit_transform(train_title,train_desc).toarray()
    X_test=cv.transform(test_title.map(str)+test_desc.map(str)).toarray()
    y=le.fit_transform(train_y)
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',random_state=42)
    classifier.fit(X_train, y)
    return le.inverse_transform(classifier.predict(X_test)).tolist()


group=test.groupby('category')



y_pred={}
for name,grp in group:
    
   
    if name in 'Autos & Vehicles':
        y_pred[name]=evaluate(autos.title,autos.description,autos.sub_category,grp.title,grp.description)
        
    elif name=='Comedy':
        y_pred[name]=evaluate(comedy.title,comedy.description,comedy.sub_category,grp.title,grp.description)
        
    elif name=='Education':
        y_pred[name]=evaluate(education.title,education.description,education.sub_category,grp.title,grp.description)
        
    elif name=='Entertainment':
        y_pred[name]=evaluate(entertainment.title,entertainment.description,entertainment.sub_category,grp.title,grp.description)
        
    elif name=='Film & Animation':
        y_pred[name]=evaluate(film.title,film.description,film.sub_category,grp.title,grp.description)
        
    elif name=='Gaming':
        y_pred[name]=evaluate(gaming.title,gaming.description,gaming.sub_category,grp.title,grp.description)
        
    elif name=='Howto & Style':
        y_pred[name]=evaluate(how.title,how.description,how.sub_category,grp.title,grp.description)
        
    elif name=='Music':
        y_pred[name]=evaluate(music.title,music.description,music.sub_category,grp.title,grp.description)
        
    elif name=='News & Politics':
        y_pred[name]=evaluate(news.title,news.description,news.sub_category,grp.title,grp.description)
        
    elif name=='People & Blogs':
        y_pred[name]=evaluate(people.title,people.description,people.sub_category,grp.title,grp.description)
    elif name=='Pets & Animals':
        y_pred[name]=evaluate(pets.title,pets.description,pets.sub_category,grp.title,grp.description)
    elif name=='Science & Technology':
        y_pred[name]=evaluate(science.title,science.description,science.sub_category,grp.title,grp.description)  
    elif name=='Sports':
        y_pred[name]=evaluate(sports.title,sports.description,sports.sub_category,grp.title,grp.description)
    elif name=='Travel & Events':
        y_pred[name]=evaluate(travel.title,travel.description,travel.sub_category,grp.title,grp.description)
    elif name=='Nonprofits & Activism':
        y_pred[name]=evaluate(non_profits.title,non_profits.description,non_profits.sub_category,grp.title,grp.description)

test['sub_category']=''
for name in y_pred.keys():
    index=group.get_group(name).index.tolist()
    for i in range(len(index)):
        test.loc[index[i],'sub_category']=y_pred[name][i]
test.to_csv('D:/Users/Dell/Documents/aml/sub_category.csv',index=False)
test.to_json('D:/Users/Dell/Documents/aml/sub_category.json',orient='values')
print('1')  

##########################
    

from sklearn.metrics import classification_report

def evaluate1(train_title, train_desc, train_y, test_title, test_desc, test_y):
    cv = CountVectorizer(max_features=5000)
    le = LabelEncoder()
    X_train = cv.fit_transform(train_title + ' ' + train_desc).toarray()
    X_test = cv.transform(test_title.map(str) + ' ' + test_desc.map(str)).toarray()
    y_train = le.fit_transform(train_y)
    
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred).tolist()
    
    # Calculate metrics
    report = classification_report(test_y, y_pred_labels, output_dict=True, zero_division=1)

    precision_macro = report['macro avg']['precision']
    
    return precision_macro

# Usage:
test_y = test['sub_category']
accuracy_macro = evaluate1(autos.title, autos.description, autos.sub_category, test.title, test.description, test_y)

accuracy_percentage = accuracy_macro * 100

print("Accuracy: {:.2f}%".format(accuracy_percentage))
