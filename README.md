# Understanding MMA Fighting Archetypes for Promotion Roster Building and Event Development

## Problem Statement
Mixed-Martial Arts (MMA) is often referred to as the one of the fastest growing sports in the world, however it has yet to leverage the power of statistics and predictive modeling.
Organizations and teams in more staple sports, such as football or basketball, have recently been investing in data science and machine learning to help improve their
odds of winning. Wheteher it be by allowingthe coaching staff to understand the flucuations in win probablity for in game decisions or just compiling data on opposing team players 
to understand even their minuscule tendencies, sport organzations have found the insights discovered through computational analytics to be a competitive advantage. 

For this project our team will look at the recorded data for MMA bouts in the sports largest promotion, the Ultimate Fighting Championship (UFC), ranging back from 1994 to 2021 in order to:
1. Understand the unique fighting archetypes across the male weight classes from featherweight through heavyweight and explore each archetype of each weight class to understand their corresponding proficiencies/deficiencies and determine if an ideal fighting style can be derived from the group
2. Explore the capability of predicting champions based on the fighter's overall statistical attributes, records, and fighting archetype from step 1.

Results of this analysis seeks to help both athletes and the promotion. Insights aim to allow trainers to better develop their fighters, while on a more macro standpoint promotions can
be more strategic in how they develop their rosters and construct their events.

## Dataset and Preprocessing
Datasets are from the following sources:\
[UFC - Fight Historical Data From 1993 to 2021](https://www.kaggle.com/datasets/rajeevw/ufcdata)\
[Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset)

The statistics in the sport of MMA are recorded for each individual bout. Statistics for each fighter and their oppostion is generalized within two main categories: standup fighting and ground fighting.
Stnandup fighting mainly consists of strikes (i.e. punches, kicks, elbows, etc.) landed by the fighter to the head, body, or legs of their opponent while ground fighting mainly consists of the number
of takedowns a fighter successfully completed, the total amount of time a fighter was able to control their opponent on the ground, the number of strikes landed on the opponent whille on the ground, and
the number of submissions attempted by the fighter as an example. In addition to the features present in the datasets, feature engineering will be conducted to explore the possibility of including
interaction variables in the models amongst the provided variables. The final dataset from the aggreation of these dtwo tables contained 1,645 unique fighters across the 9 weight classes with 61 corresponding attributes.

Example of features:
* AVG KD is number of knockdowns
* AVG SIG_STR is no. of significant strikes 'landed of attempted'
* SIG_STR_pct is significant strikes percentage
* AVG TOTAL_STR is total strikes 'landed of attempted'
* AVG TD is no. of takedowns
* Stance is the stance of the fighter (orthodox, southpaw, etc.)
* Height_cms is the height in centimeter
* Reach_cms is the reach of the fighter (arm span) in centimeter
* Weight_lbs is the weight of the fighter in pounds (lbs)
* current_lose_streak is the count of current concurrent losses of the fighter
* current_win_streak is the count of current concurrent wins of the fighter
* draw is the number of draws in the fighter's ufc career
* wins is the number of wins in the fighter's ufc career
* losses is the number of losses in the fighter's ufc career
* total_rounds_fought is the average of total rounds fought by the fighter
* win_by_Decision_Majority is the number of wins by majority judges decision in the fighter's ufc career
* win_by_Decision_Split is the number of wins by split judges decision in the fighter's ufc career
* win_by_Decision_Unanimous is the number of wins by unanimous judges decision in the fighter's ufc career
* win_by_KO/TKO is the number of wins by knockout in the fighter's ufc career
* win_by_Submission is the number of wins by submission in the fighter's ufc career
* win_by_TKO_Doctor_Stoppage is the number of wins by doctor stoppage in the fighter's ufc career
