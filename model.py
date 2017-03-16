import pandas as pd
import math
import random
import csv
from sklearn import cross_validation, linear_model
import itertools
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

team_stats = {}
X = []
y = []
prediction_data = []
avg_period = 10 # the number of games to average statistics over
all_stats_df = pd.read_csv("proc_data/FinalStats.csv")

stat_fields = ["score", "fgm%", "fgm3%", "ftm%", "+/-", "or", "dr", "ast", "blk", "elo", "sor_rk",
               "non_conf_sos", "bpi", "q_win", "q_loss"]


for i in range(2007, 2018):
    team_stats[i] = {}

with open("pickled_data/team_dict.pickle", "r") as td_file:
    team_dict = pickle.load(td_file)
td_file.close()


def update_stats(season, team, fields):
    if team not in team_stats[season]:
        team_stats[season][team] = {}
        
    for key,value in fields.items():
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []
        
        if len(team_stats[season][team][key]) >= avg_period: 
            team_stats[season][team][key].pop()
               
        team_stats[season][team][key].append(value)

               
def get_stat_avg(season, team, field):
    # if elo we don't want to return the avg, just the most recent.
    try:
        stat = team_stats[season][team][field]
        if field == "elo":
            return team_stats[season][team][field][-1]
        else:
            return sum(stat)/float(len(stat))
    except:
        return 0


def build_season_data():
    for index, row in all_stats_df.iterrows():
        # skip matchups where we dont have stats yet
        skip = 0
        
        team1_features = []
        team2_features = []
        
        for field in stat_fields:
            team1_stat = get_stat_avg(row["Season"], row["Wteam"], field)
            team2_stat = get_stat_avg(row["Season"], row["Lteam"], field)
            
            if team1_stat is not 0 and team2_stat is not 0:
                team1_features.append(team1_stat)
                team2_features.append(team2_stat)
            else:
                skip = 1
            
        if skip == 0: # make sure stats exist
            # randomly order team stats on the right or left so 
            # that model can recognize a pattern for prediction

            if random.random() > 0.5:
                X.append(team1_features + team2_features)
                y.append(0)
            else:
                X.append(team2_features + team1_features)
                y.append(1)
            
        stat1_fields = {
            'score': row['Wscore'],
            'fgm%': row['Wfgm%'],
            'fgm3%': row['Wfgm3%'],
            'ftm%': row['Wftm%'],
            '+/-': row['W_+/-'],
            'or': row['Wor'],
            'dr': row['Wdr'],
            'ast': row['Wast'],
            'blk': row['Wblk'],
            'elo': row['w_elo_before_game'],
            'sor_rk': row['W_SOR_RK'],
            'non_conf_sos': row['W_non_conf_sos'],
            'bpi': row['W_BPI'],
            'q_win': row['W_Q_wins'],
            'q_loss': row['W_Q_losses']      
        }

        stat2_fields = {
            'score': row['Lscore'],
            'fgm%': row['Lfgm%'],
            'fgm3%': row['Lfgm3%'],
            'ftm%': row['Lftm%'],
            '+/-': row['L_+/-'],
            'or': row['Lor'],
            'dr': row['Ldr'],
            'ast': row['Last'],
            'blk': row['Lblk'],
            'elo': row['l_elo_before_game'],
            'sor_rk': row['L_SOR_RK'],
            'non_conf_sos': row['L_non_conf_sos'],
            'bpi': row['L_BPI'],
            'q_win': row['L_Q_wins'],
            'q_loss': row['L_Q_losses']      
        }
        update_stats(row['Season'], row['Wteam'], stat1_fields)
        update_stats(row['Season'], row['Lteam'], stat2_fields)
        
    return X, y


def predict_winner(team1, team2, model, season, stat_fields, proba):
    features = []
    for stat in stat_fields:
        features.append(get_stat_avg(season, team1, stat))
    
    for stat in stat_fields:
        features.append(get_stat_avg(season, team2, stat))
    
    if proba:
        return model.predict_proba(features)
    else:
        return model.predict(features)


X, y = build_season_data()

# train logistic regression model
print "Fitting on {} samples.".format(len(X))
model = linear_model.LogisticRegression(solver='sag')

print("Doing cross-validation.")
print(cross_validation.cross_val_score(model, X, y, cv=10, scoring='accuracy', n_jobs=-1).mean())

model.fit(X, y)

# train and test some other models
train_X = X[:26000]
train_y = y[:26000]

test_X = X[26000:]
test_y = y[26000:]

# neural network classifier
nn_clf = MLPClassifier(solver='adam', alpha=1e-5)
nn_clf.fit(train_X, train_y)
print nn_clf.score(test_X, test_y)

rf_clf = RandomForestClassifier(n_estimators=50)
rf_clf.fit(train_X, train_y)
print rf_clf.score(test_X, test_y)

all_seeds_df = pd.read_csv("data/TourneySeeds.csv")
seeds2017_df = all_seeds_df[all_seeds_df["Season"] == 2017]
seeds2017_df.head()

# get the teams playing in this years tournament and find all the possible matchups
t_year = 2017
t_teams = seeds2017_df["Team"]
possible_matchups = []
for subset in itertools.combinations(t_teams, 2):
    possible_matchups.append(subset)


# decided to use random forest classifier here, can use any other one by swapping out model param in predict_winner function
for matchup in possible_matchups:
    team_1 = matchup[0]
    team_2 = matchup[1]
    prediction = predict_winner(team_1, team_2, rf_clf, t_year, stat_fields, proba=True)
    label = str(t_year) + '_' + str(team_1) + '_' + str(team_2)
    prediction_data.append([label, prediction[0][0]])


results_list = []

for pred in prediction_data:
    parts = pred[0].split('_')
    # Order them properly.
    if pred[1] > 0.5:
        winning = int(parts[1])
        losing = int(parts[2])
        proba = pred[1]
    else:
        winning = int(parts[2])
        losing = int(parts[1])
        proba = 1 - pred[1]
    results_list.append(
        [
            '%s beats %s: %f' %
            (team_dict[winning], team_dict[losing], proba)
        ]
    )


with open('predictions.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(results_list)


# Method 2: run simulations to see who would win the most teams
def build_tourneydict():
    tourney_dict = {}
    for index, row in seeds2017_df.iterrows():
        if "a" in row["Seed"]:
            row["Seed"] = row["Seed"][:3]
        tourney_dict[row["Seed"]] = {}
        tourney_dict[row["Seed"]]["Team"] = row["Team"]

    return tourney_dict


slots_df = pd.read_csv("data/TourneySlots.csv")
slots2017_df = slots_df[slots_df["Season"] == 2017]

team_wins = {}
team_wins = team_dict.copy()

for key,val in team_wins.iteritems():
    team_wins[key] = 0


# simulate the tournament any number of times to see who wins the most games
def run_simulation(iterations):
    tourney_copy = build_tourneydict()
    for i in range(0,iterations):
        for index, row in slots2017_df.iterrows():
            if "a" in row["Strongseed"] or "b" in row["Weakseed"]:
                row["Strongseed"] = row["Strongseed"][:3]
                row["Weakseed"] = row["Weakseed"][:3]

            team1 = tourney_copy[row["Strongseed"]]["Team"]
            team2 = tourney_copy[row["Weakseed"]]["Team"]
            
            win_prob = predict_winner(team1, team2, model, 2017, stat_fields, proba=True) 
            rand = float(random.randint(0,100))
            
            if (win_prob[0][0] * 100) > rand:
                winner = team1
                tourney_copy[row["Slot"]] = tourney_copy.pop(row["Strongseed"])
                team_wins[team1] += 1
            else:
                winner = team2
                tourney_copy[row["Slot"]] = tourney_copy.pop(row["Weakseed"])
                team_wins[team2] += 1

            slots2017_df.set_value(index, "win_prob", win_prob[0][0])           
                        
            if row["Slot"] == "R6CH":
                tourney_copy = build_tourneydict()


run_simulation(10000)   

wins_list = [(team_dict[key], val) for key, val in team_wins.iteritems()]

with open('team_wins.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(wins_list)          
