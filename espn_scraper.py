import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import pickle


with open("pickled_data/team_dict.pickle", "rb") as dict_file:
    team_dict = pickle.load(dict_file)
dict_file.close()

url = "http://www.espn.com/mens-college-basketball/bpi/_/season/{}/page/{}/view/{}"
num_pages = 16
seasons = [i for i in range(2008, 2018)]


# scrape data from espn into dataframe
def get_stats(statstype):
    for season in seasons:
        for num in range(1, num_pages):
            formatted_url = url.format(str(season), str(num), statstype)
            if season == 2008 and num == 1:
                df_list = pd.read_html(formatted_url)
                stat_df = df_list[1]
                stat_df["Season"] = season
            else:
                try:
                    df_list = pd.read_html(formatted_url)
                    df_to_append = df_list[1]
                    df_to_append["Season"] = season
                    stat_df = stat_df.append(df_to_append, ignore_index=True)
                except IndexError:
                    # Depending on the season, pages can have less than 15 pages
                    print formatted_url
    
    return stat_df


# get dataframe with espn's team resume data
resume_df = get_stats("resume")
print resume_df.head()

# remove the short form team name from the end of string
resume_df["TEAM"] = resume_df["TEAM"].apply(lambda x: re.sub(r"(?<=[a-z])[A-Z]+$","", x))
resume_df.drop(["CONF", "Seed", "W-L", "SOR S-Curve"], axis=1, inplace=True)
print resume_df.head()

# get dataframe with espn's bpi stats
bpi_df = get_stats("bpi")
bpi_df["TEAM"] = bpi_df["TEAM"].apply(lambda x: re.sub(r"(?<=[a-z])[A-Z]+$","", x))
bpi_df.drop(["CONF", "W-L", "7-Day RK CHG"], axis=1, inplace=True)
bpi_df.tail()

resume_df.sort_values(by=["Season", "TEAM"], inplace=True)
bpi_df.sort_values(by=["Season", "TEAM"], inplace=True)

espn_df = resume_df.merge(bpi_df, on=["TEAM", "Season"])
espn_df.head()

team_list = list(team_dict.values())

for index, row in espn_df.iterrows():
    for team in team_list:
        if team in row["TEAM"]:
            espn_df.set_value(index,'Teamname',team)
        else:
            row["Teamname"] = "fail"


cols = ["Season","Teamname", "SOR RK", "SOS RK", "Non-Conf SOS RK", "Qual W-L", "BPI"]
espn_df = espn_df[cols]

Q_cols = pd.DataFrame(espn_df["Qual W-L"].str.split('-',1).tolist(), columns = ['Q_wins','Q_losses'])

ESPN_df = espn_df.join(Q_cols)
ESPN_df.drop("Qual W-L", axis=1, inplace=True)
