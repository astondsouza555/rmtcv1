api_key='AIzaSyBl2UGNNR6xKP4-jMW5KlCTGbXuLwJ-Hhg'

from googleapiclient.discovery import build

youtube=build('youtube','v3',developerKey=api_key)

import pandas as pd
ca=pd.read_csv('D:/Users/Dell/Documents/aml/category_list.csv')

#This function takes a category ID as input, 
#makes API requests to fetch videos belonging to that category, and 
#returns the retrieved video items.
def videos(catId):
    req1=youtube.search().list(part='id',type='video',videoCategoryId=catId,maxResults=20) #only videoId is retrieved
    res1=req1.execute()
    req=youtube.videos().list(part='snippet,statistics',id=','.join([item['id']['videoId'] for item in res1['items']])) # These video IDs are then used to fetch detailed information about each video.
    res=req.execute()

    return res['items']

values=[]
# It loops over each category in the DataFrame ca where the 'assignable' column is True.
for i in ca[ca['assignable']==True]['id'].tolist():
# For each category, it calls the videos function to fetch video data. 
#It extracts relevant information such as video ID, title, description, view count, 
#like count, dislike count, comment count, category, image URL, and tags.    
    for item in videos(i):
        video_id=item['id']
        title=item['snippet']['title']
        desc=item['snippet']['description']
        views=item['statistics'].get('viewCount',0)
        likes=item['statistics'].get('likeCount',0)
        dislikes=item['statistics'].get('dislikeCount',0)
        comments=item['statistics'].get('commentCount',0)
        cat=ca.loc[ca['id']==int(item['snippet']['categoryId']),'title'].iloc[0]
        image=item['snippet']['thumbnails']['default']['url']
        tags=','.join(item['snippet'].get('tags',''))
        values.append((video_id,title,desc,views,likes,dislikes,comments,cat,image,tags))
# It stores the fetched video data into a pandas DataFrame named data.
data=pd.DataFrame(values,columns=['video_id','title','description','viewcount','likecount','dislikecount','commentcount','category','image','tags'])

data.to_csv('D:/Users/Dell/Documents/aml/auto.csv',index=False)
data.to_json('D:/Users/Dell/Documents/aml/auto.json',orient='values')

print('1')