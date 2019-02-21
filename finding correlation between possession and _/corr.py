import pandas as pd

df = pd.read_csv('FootballEurope.csv')
interesting_columns = ['homePossessionFT', 'awayPossessionFT', 'homeGoalFT','awayGoalFT', 'homeShotsTotalFT', 'awayShotsTotalFT']
df.dropna(subset=interesting_columns, inplace=True)

print(df[interesting_columns].corr())
