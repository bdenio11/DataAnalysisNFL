import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt

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
