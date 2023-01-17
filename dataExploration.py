import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

games = pd.read_csv("../../NFLBigDataBowl2022/games.csv")
players = pd.read_csv("../../NFLBigDataBowl2022/players.csv")
plays = pd.read_csv("../../NFLBigDataBowl2022/plays/plays.csv")
pffScoutingData = pd.read_csv("../../NFLBigDataBowl2022/PFFScoutingData/PFFScoutingData.csv")
tracking2018 = pd.read_csv("../../NFLBigDataBowl2022/tracking2018/tracking2018.csv")

games2018 = games[games.season == 2018]

plays2018 = pd.merge(games2018, plays, on='gameId') # All plays of 2018
plays2018_extraPoint = plays2018[plays2018['specialTeamsPlayType'] == 'Extra Point']

plays2018_extraPoint_results = plays2018_extraPoint['specialTeamsResult'].value_counts()[
    ['Kick Attempt Good', 'Kick Attempt No Good','Non-Special Teams Result', 'Blocked Kick Attempt']]

plays2018_extraPoint_graph = sns.barplot(x=plays2018_extraPoint_results.index, y=plays2018_extraPoint_results)
plays2018_extraPoint_graph.bar_label(plays2018_extraPoint_graph.containers[0])

plays2018_extraPoint_graph.set_xticklabels(plays2018_extraPoint_graph.get_xticklabels(), rotation=40, ha="right")
plt.pyplot.title('Extra Points Results')
plt.pyplot.ylabel('Count')

corrMap = plays2018_extraPoint.drop(columns=['gameId', 'season', 'week', 'playId', 'down', 'yardsToGo', 
    'kickLength', 'kickReturnYardage','playResult', 'passResult'])
pd.options.display.max_columns = None
corrMap.head()
corr_matrix = corrMap.corr()

corr_matrix = corr_matrix[(corr_matrix > 0.2)]

sns.heatmap(corr_matrix, annot=True)


X = plays2018_extraPoint[['kickerId','kickBlockerId', 'preSnapHomeScore', 'preSnapVisitorScore']]
X.fillna(-1, inplace=True)

X2 = plays2018_extraPoint[['kickerId','kickBlockerId', 'preSnapHomeScore', 'preSnapVisitorScore',
                           'week', 'gameTimeEastern', 'gameClock']]

X2['gameClock'] = X2['gameClock'].apply(lambda x: float(x.split(':')[0]) * 60 + float(x.split(':')[1])) 
X2['gameTimeEastern'] = X2['gameTimeEastern'].apply(lambda x: float(x.split(':')[0]) * 60 + float(x.split(':')[1])) 

X2 = X2.fillna(-1)
# Define the dependent variable
y = plays2018_extraPoint['specialTeamsResult']

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2)


# create the decision tree classifier
clf = DecisionTreeClassifier()

# train the classifier on the training data
clf.fit(X_train, y_train)

# make predictions on the test set
y_pred = clf.predict(X_test)

# evaluate the performance of the classifier
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(100,20))

#plot_tree(clf, feature_names=X_train.columns, filled=True, rounded=True, class_names=clf.classes_, values=values)
plot_tree(clf, feature_names=X_train.columns, filled=True, rounded=True, class_names=clf.classes_);
plt.show()

print("Feature importances: ", clf.feature_importances_)
print("classes", clf.feature_names_in_)

