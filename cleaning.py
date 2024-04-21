import pandas as pd
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
a=0.0
v=0.0
if os.path.exists('D:/Users/Dell/Documents/aml/auto.csv'):
	a=os.path.getmtime('D:/Users/Dell/Documents/aml/auto.csv')
if os.path.exists('D:/Users/Dell/Documents/aml/video.csv'):
	v=os.path.getmtime('D:/Users/Dell/Documents/aml/video.csv')
if a==max(a,v):
	name='auto'
else:
	name='video'
data=pd.read_csv('D:/Users/Dell/Documents/aml/'+name+'.csv')
print(name) # why video
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# it preprocesses the text data in the DataFrame:

# Converts the 'title' and 'description' columns to lowercase.
# Removes punctuation from 'title' and 'description' columns.
# Strips leading and trailing whitespaces from 'title' and 'description' columns.
# Tokenizes 'title' and 'description' columns into lists of words.
# Removes stopwords from 'title' and 'description' columns.
# Lemmatizes words in 'title' and 'description' columns (reducing them to their base or dictionary form, e.g., "running" to "run").
# Joins the lists of words back into strings for 'title' and 'description' columns.

data.dropna(inplace=True)
data.reset_index(drop=True,inplace=True)
data['title'] = data['title'].map(lambda x: x.lower())
data['description'] = data['description'].map(lambda x: x.lower())
data['title']  = data['title'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
data['description']  = data['description'].map(lambda x: x.translate(x.maketrans('', '', string.punctuation)))
data['title'] = data['title'].map(lambda x: x.strip())
data['description'] = data['description'].map(lambda x: x.strip())
data['title'] = data['title'].map(lambda x: word_tokenize(x))
data['description'] = data['description'].map(lambda x: word_tokenize(x))
data['title'] = data['title'].map(lambda x: [word for word in x if word.isalpha()])
data['description'] = data['description'].map(lambda x: [word for word in x if word.isalpha()])
stop_words = set(stopwords.words('english'))
data['title'] = data['title'].map(lambda x: [w for w in x if not w in stop_words])
data['description'] = data['description'].map(lambda x: [w for w in x if not w in stop_words])
lem = WordNetLemmatizer()
data['title'] = data['title'].map(lambda x: [lem.lemmatize(word,"v") for word in x])
data['description'] = data['description'].map(lambda x: [lem.lemmatize(word,"v") for word in x])
data['title'] = data['title'].map(lambda x: ' '.join(x))
data['description'] = data['description'].map(lambda x: ' '.join(x))
data.sort_values("video_id", inplace = True) 
  
# dropping ALL duplicte values 
data.drop_duplicates(subset ="video_id", 
                     keep = False, inplace = True) 
data.reset_index(inplace=True,drop=True)
data.to_csv('D:/Users/Dell/Documents/aml/clean.csv',index=False)
data.to_json('D:/Users/Dell/Documents/aml/clean.json',orient='values')
print('1')