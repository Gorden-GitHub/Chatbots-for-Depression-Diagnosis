# Chatbots-for-Depression-Diagnosis
Data Science for Text Analytics Project

## 1 General Submission Guildelines
+ **Title:** Chatbots for Depression Diagnosis
+ **Team Members:** \
Changjing Hu, 3707220, Scientific computing\
Xinyu Liang, 3770373, Scientific computing\
Guangdeng Liang, 3769325, Data and computer science\
Shengzhe Xu, 3769534, Data and computer science
+ **Mail Addresses:**\
changjing.hu@stud.uni-heidelberg.de\
xinyu.liang@stud.uni-heidelberg.de\
gorden.ggm@gmail.com\
xushengzhe2021de@163.com
+ **Existing Code Fragments:**
+ **Utilized libraries:**

  https://github.com/Gorden-GitHub/Chatbots-for-Depression-Diagnosis/blob/main/requirements.txt

+ **Contributions:**
  - Guangdeng Liang was responsible for adding the database details and building the model.
  - Changjing Hu was responsible for writing the proposal and part of the milestones.
  - Xinyu Liang was responsible for researching the data analysis model and writing part of the milestones.
  - Shengzhe Xu was responsible for researching and learning the front-end of the website.

+ **Uploading for other team members:**

## 2 Project State
+ **Planning State:**
<img width="500" alt="image" src="https://user-images.githubusercontent.com/69336330/206926245-63ce01e0-e9d8-4532-8c75-7215a20dfd2f.png">

+ **Future Planning:**

  In the second stage, the main plan is to complete the construction of the BERT model and chatbot model, which is done by Guangdeng Liang. Here Changjing Hu also introduces the LSTM model for the psychological prediction of users. After all the models are prepared, Shengzhe Xu will package these models together for the future webpage design. Xinyu Liang will also provide ideas for data analysis.

+ **High-level Architecture Description:**

  - Code Project (BERT)
  
    <img width="500" alt="image" src="https://user-images.githubusercontent.com/69336330/206926664-3b271bd6-2c29-4e20-bdf3-6ae0f849f088.png">

  - Processing Pipeline:\
     a) Regular Expressions\
     b) Tokenization\
     c) Normalization\
     d) Lemmatization\
     e) Stemming\
     f) Stop Words

+ **Experiments:**
  - BERT model\
    Here input four complete sentences as test example: \
    a)  “depress mental retard movement limit exercis mechan”; \
    b)  “orri like veri much even group photo”;\
    c)  “easier remov world”;\
    d)  “want die”;\
    <img width="1200" alt="image" src="https://user-images.githubusercontent.com/69336330/206926751-f7af6144-3dc2-4486-9c27-153f3bc02a9b.png">
    
    After the input sentences are preprocessed, they are handed over to the BERT model for emotional judgment, and finally, the possibility of depression is obtained respectively. The test results are shown in the table:
    
    |Sentences: |
    |------|
    |“depress mental retard movement limit exercis mechan.”|
    |“orri like veri much even group photo.”|
    |“easier remov world.”|
    |“want die.”|
    
    Are you depressed?
    |  No   | Yes  |
    |  ----  | ----  |
    | 0.01186199 | 0.988138 |
    | 0.97349596  | 0.02650399 |
    | 0.8946904 | 0.10530968 |
    |0.0232484|0.97675157|

  - LSTM model
  
    Users participating in the test need to have a daily conversation with the chatbot. After collecting the 30-day chat records of each user and judging the possibility of depression in each sentence through the BERT model, we get the raw data and calculate the max, min, mean, and variance of the probability of depression for the conversation record each day. Then, they are stored in a separate file for each user.\
    <img width="500" alt="image" src="https://user-images.githubusercontent.com/69336330/206927491-3da50aae-8824-4115-80ca-8cc2e6818e6a.png">
  
    In order to judge the tendency toward depression, we introduce the long short-term memory network (LSTM model) and make predictions by it. Mean squared error for each epoch is shown in the figure.\
    <img width="500" alt="image" src="https://user-images.githubusercontent.com/69336330/206927512-f0955f45-9088-476d-bf85-f3f3c3ef650f.png">
  
    Here the data from the first 24 days will be used as the training set, and the data from the last 6 days will be used as the test set. The results are shown below:\
    <img width="500" alt="image" src="https://user-images.githubusercontent.com/69336330/206927532-4c7612d7-6676-49d7-8c12-edd98caabd36.png">
    
## 3 Data Analysis
+ **Data Sources:**

  - Data source website: https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned.
  - The data is stored as 'Dataset\depression_dataset_reddit_cleaned.csv'.
  - The original data was obtained from the Reddit website through the web scrapping Subreddits technique. The data is only available in English. It focuses on mental health categories. The dataset has a total of 7650 available data items.

+ **Preprocessing:**

  - First, the word type reduction function nltk.WordNetLemmatizer() was used to remove the affixes from the words and reduce the words to their base words.
  - SnowballStemmer() function was used to extract the stemming of the English text.
  - Afterwards, stopwords are processed to remove words such as "http", "twitpic", "com", "tinyurl", "co", "wa" and stopwords from the "english" collection.
  - For example, “I want to die" would be "want die" after the above three processes.
  
+ **BasicStatistics:**

  - Use the pandas software library in python.
  - Read the data inside the 'Dataset\depression_dataset_reddit_cleaned.csv' file.
  - Plot the bar graph against the 'is_depression' value.
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/69336330/206929584-fbaeecdb-fb2a-49b5-bbf2-cb9d81468632.png"> 
  
  - Define data with an 'is_depression' value of 1 as df_depression. define data with an 'is_depression' value of 0 as df_non_depression. facilitate The data can be analysed later.
  - Calculate the maximum and average number of words in the sentences of the depression category and the non-depression category respectively.
  - Extract the 10 most common triads in the dataset and their frequency of occurrence.
  - According to the results, ["http", "twitpic", "com", "tinyurl", "co"] should be added to the Stop_words.
  <img width="191" alt="image" src="https://user-images.githubusercontent.com/69336330/206930593-a85f0bd9-c783-43be-970c-2b9392840541.png">
  
  - Using WordCloud() to plot word clouds for depressed and non-depressed messages. Based on the results, ["wa"] should be added to Stop_words.
  ![image](https://user-images.githubusercontent.com/69336330/206929889-abaeaba4-69c1-454a-8942-79f5c8b378ea.png)\
  ![image](https://user-images.githubusercontent.com/69336330/206929981-1e176486-4ba0-47ec-9cdc-af0468bbc1e5.png)

+ **Examples:**

  **Input:**
  
  we understand that most people who reply immediately to an op with an invitation to talk privately mean only to help but this type of response usually lead to either disappointment or disaster it usually work out quite differently here than when you say pm me anytime in a casual social context we have huge admiration and appreciation for the goodwill and good citizenship of so many of you who support others here and flag inappropriate content even more so because we know that so many of you are struggling yourselves we re hard at work behind the scene on more information and resource to make it easier to give and get quality help here this is just a small start our new wiki page explains in detail why it s much better to respond in public comment at least until you ve gotten to know someone it will be maintained at r depression wiki private contact and the full text of the current version is below summary anyone who while acting a a helper invite or accepts private contact i e pm chat or any kind of offsite communication early in the conversion is showing either bad intention or bad judgement either way it s unwise to trust them pm me anytime seems like a kind and generous offer and it might be perfectly well meaning but unless and until a solid rapport ha been established it s just not a wise idea here are some point to consider before you offer or accept an invitation to communicate privately by posting supportive reply publicly you ll help more people than just the op if your response are of good quality you ll educate and inspire other helper the 9 90 rule http en wikipedia org wiki rule internet culture applies here a much a it doe anywhere else on the internet people who are struggling with serious mental health issue often justifiably have a low tolerance for disappointment and a high level of ever changing emotional need unless the helper is able to make a 00 commitment to be there for them in every way for a long a necessary offering a personal inbox a a resource is likely to do more harm than good this is why mental health crisis line responder usually don t give their name and caller aren t allowed to request specific responder it s much healthier and safer for the caller to develop a relationship with the agency a a whole analogously it s much safer and healthier for our ops to develop a relationship with the community a a whole even trained responder are generally not allowed to work high intensity situation alone it s partly about availability but it s mostly about wider perspective and preventing compassion fatigue if a helper get in over their head with someone whose mental health issue including suicidality which is often comorbid with depression escalate in a pm conversation it s much harder for others including the r depression and r suicidewatch moderator to help contrary to common assumption moderator can t see or police pm in our observation over many year the people who say pm me the most are consistently the one with the least understanding of mental health issue and mental health support we all have gap in our knowledge and in our ability to communicate effectively community input mitigates these limitation there s no reason why someone who s truly here to help would want to hide their response from community scrutiny if helper are concerned about their own privacy keep in mind that self disclosure when used supportively is more about the feeling than the detail and that we have no problem here with the use of alt throwaway account and have no restriction on account age or karma we all know the internet is used by some people to exploit or abuse others these people do want to hide their deceptive and manipulative response from everyone except their victim there are many of them who specifically target those who are vulnerable because of mental health issue if a helper invite an op to talk privately and give them a good supportive experience they ve primed that person to be more vulnerable to abuser this sort of cognitive priming tends to be particularly effective when someone s in a state of mental health crisis when people rely more on heuristic than critical reasoning if ops want to talk privately posting on a wide open anonymous forum like reddit might not be the best option although we don t recommend it we do allow ops to request private contact when asking for support if you want to do this please keep your expectation realistic and to have a careful look at the history of anyone who offer to pm before opening up to them

  **Output:**
  
  ![image](https://user-images.githubusercontent.com/69336330/206931711-973c87b5-3b40-4c7b-83e1-7ed6086f21cf.png)










