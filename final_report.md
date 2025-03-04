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
### <ins>Dataset</ins>
Datasets are from the following sources:\
[UFC - Fight Historical Data From 1993 to 2021](https://www.kaggle.com/datasets/rajeevw/ufcdata)\
[Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset)

The statistics in the sport of MMA are recorded for each individual bout. Statistics for each fighter and their oppostion is generalized within two main categories: standup fighting and ground fighting.
Stnandup fighting mainly consists of strikes (i.e. punches, kicks, elbows, etc.) landed by the fighter to the head, body, or legs of their opponent while ground fighting mainly consists of the number
of takedowns a fighter successfully completed, the total amount of time a fighter was able to control their opponent on the ground, the number of strikes landed on the opponent whille on the ground, and
the number of submissions attempted by the fighter as an example. In addition to the features present in the datasets, feature engineering will be conducted to explore the possibility of including
interaction variables in the models amongst the provided variables. The final dataset from the aggreation of these dtwo tables contained 1,645 unique fighters across the 9 weight classes with 61 corresponding attributes.

Example of features:\
AVG KD: average number of knockdowns achieved\
AVG SIG_STR: average number of significant strikes landed / significant strikes thrown\
SIG_STR_pct: significant strikes landed / significant strikes thrown\
AVG TOTAL_STR: total strikes\
AVG TD: number of takedowns\
Stance: is the stance of the fighter (orthodox, southpaw, etc.)\
Height_cms: is the height in centimeter\
Reach_cms: is the reach of the fighter (arm span) in centimeter\
Weight_lbs: is the weight of the fighter in pounds (lbs)\
current_lose_streak: is the count of current concurrent losses of the fighter\
current_win_streak: is the count of current concurrent wins of the fighter\
draws: is the number of draws in the fighter's ufc career\
wins: is the number of wins in the fighter's ufc career\
losses: is the number of losses in the fighter's ufc career\
total_rounds_fought: is the average of total rounds fought by the fighter\
win_by_Decision_Majority: is the number of wins by majority judges decision in the fighter's ufc career\
win_by_Decision_Split: is the number of wins by split judges decision in the fighter's ufc career\
win_by_Decision_Unanimous: is the number of wins by unanimous judges decision in the fighter's ufc career\
win_by_KO/TKO: is the number of wins by knockout in the fighter's ufc career\
win_by_Submission: is the number of wins by submission in the fighter's ufc career\
win_by_TKO_Doctor_Stoppage: is the number of wins by doctor stoppage in the fighter's ufc career\

### <ins>Preprocessing</ins>
The following preprocessing and aggregation steps were taken to combine the datasets into a final dataset

* **Feature Engineering**: To reduce multi-collinearity issues for the Logistic Regression model, certain features were combined (ex. combining "avg_TD_landed" with "avg_TD_attempted" into "TD_landed_Percentage"). General "counts" were calculated for fighters as well such as their total number of fights and win streaks.\

* **Handling Nulls**: Upon joining the datasets on its primary key (fighter name), there were data fields that were empty or missing upon joining. For these fighters missing important features, a manual review was conducted to see if there were ever champions and upon validation that they weren't a decision was made to remove these fighters from the final dataset. The preservation of champions was conducted to preserve the already small imbalanced population size of the target variable.

* **Dropping Features**: "Age" and "time" related features were drop from the dataset due to the granularity of our dataset being at the fighter level as opposed to the fighter at a certain point of time (i.e. Fighter A at the age of 25 vs 30 years of age).

* **Dropping Figherss**: A minimum of three fights were required of each fighter to be included in the analysis.

* **Assigning "Champion" attribute**: The "champion" attribute which acts as the target variable for the prediction model was derived by identifying if a fighter has ever once won a title fight.

The final dataset yields 1,447 uniue fighters with 48 attributes consisting of 45 numerical variables, 2 catergorical variables, and the target variable. There are a total of 119 unique champions in the dataset. 
<br>
<br>

<p align="center">
<b>Distribution of Fighters by Weight Class:</b>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/fbf41e5c-d1e7-48b0-82c8-861c08de40f1" />
</p>

## Methodology
### K-Means Clustering for Fighting Styles
In order to derive the fighting styles of each weigh class we utilized the unsupervised learning method of the K-means clustering algorithm, initialzing the model with various numbers of cluserts to find the optimal grouping. The Silhouette Scores of these grouping will help with finding the optimal grouping as such method explores the average distinctiveness of each cluster with the output clustering assignment.

$$\text{Silhouette Coefficient of Data Point(o)} = \frac{b(o)-a(o)}{max[a(o), b(o)]}$$
$$\boldsymbol{a(0)} = \text{average distance between point(o) and all the other data points within the same cluster}$$
$$\textbf{b(0)} = \text{minimum average distance between point(o) to all clusters to which it doesn't belong to}$$
$$\textbf{s(o)} = \text{silhouette coefficient of data point(o)}$$

$$\text{Silhouette Score} = \frac{b-a}{max[a,b]}$$
$$\boldsymbol{a(0)} = \text{average intra-cluster distance}$$
$$\boldsymbol{b(0)} = \text{average inter-cluster distance}$$

Once the clustering is finalized a pivot table was created to outline the characteristics of each archetype, highlight each stypes proficiencies and deficiences,
and lastly examine the number of champions each fighting style has produced, as well as that number's to the total number of fighters in that division over the total champions in that division to gauge the effectiveness of that style. The random seed will be set to the same value across all K-means clustering trials to reduce the fluctuation of results across difference sessions.

The final clustering groups will act as the fighters assigned style in the proceeding prediciton phase of the analysis. A feature which we hypthesize will improve the accuracy of the trialed models in predicting whether or not a fighter will become a champion

### Champion Prediction Model
In predicting wheter a fighter will be a champion within their weight class, a logistic regression and a foreset based model were both built. In prepartion for each weight class, the following is the population of fighters in each weight class and number of champions in their respective weight classes.

<p align="center">
<b>Population and Champion Distribution Per Weight Class</b>
</p>

<p align="center">
<img width="392" alt="Screenshot 2025-02-18 at 8 22 11 PM" src="https://github.com/user-attachments/assets/d5848089-3973-4fa3-b5e6-447f09cb8422" />
</p>

Due to the limited number of champions present in each weight class and teh mixture of gender in the lower weight classes which adds another dimension for later exlploration, our models will
primarily focus on weight calsses from featherweigh and above.

The table above also shows the class imbalance between champions and non-champions. Oversampling techniques such as Synthetic Minority Oversampling Technique (SMOTE) are also applied to the training data to be able to get more diversity out of a limited data set. By applying oversampling techniques, the training data is augmented to provide more info0rmation on the
imbalanced class (i.e. champions).

The data also went through standardiation, though this may not be necessary for random forest or regression models, there is a benefit of convergence speeds for the regression modesls, so the standardization was implemented. 

In search of the optimal hyperparameters for both the random forest, and logistic regression models, k-fold validation methods were used to train and validate teh iteration of models as the optimal hyperparameters were searched for. The cross validation value is set to 5 and used in a grid search search for the optimal hyperparametes.

Focusing on the logistic regression model, different regularization methods are looked at such as the LASSO, Ridge, and Elastic Net method.  

<p align="center">
C value: 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100 <br>
L1 ratio: 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.99, 1.0
</p>

The goal of the regularization tuning term, "C", is to correct for overfitting and simplify the model to generalize the predictions better. This is done differently inboth LASSO and Ridge Regression. LASSO penalizes the absolute value of the coefficient and sets irrelevant feature coefficeint values to 0. Ridge penalizes the size of the regression coefficient by the square of the magnitude while not reducing the feeature coefficients to 0.

LASSO can help with feature selection, but is unstable when varaibles are correlated. Ridge on the other hand can handle correlated variables but can not perform feature selection. ElasticNet is the combination of theses two regualarization methods by balancing that ration with the L1 ratio. The following are teh mathematical functions for LASSO, Ridge, and Elastic Net Regression.

<p align="left">
$$\text{LASSO} = \text{min}\frac{1}{m}\sum_{i=1}^m (y^i - \theta^T \text{x}^i + \lambda $|\theta|_1)$$
<br>
$$\text{Ridge} = \text{min}\frac{1}{m}\sum_{i=1}^m (y^i - \theta^T \text{x}^i + \lambda $|\theta|_2^2)$$
<br>
$$\text{ElasticNet} = $$
</p>

## Results and Evaluation
### Derivation of Fighting Styles
The exectuion of the K-means clustering was done by weight class nine separate times with the first initalization being made with seven clusters and incrementally increasing to 15. The starting number of seven was chosen in reference to a graphic previously published by ESPN that outlines the top seven winning fighting styles proved by the fighters tehmselves.
1. Wrestling
2. Brazilian Jiu-Jitsu
3. Boxing
4. Kickboxing
5. Muay Thai
6. Taekwondo
7. Karate

By increasing the number of initialized clusters incrementally, we believe it will allow for the algorithm to discover more distinct fighting styles from the seven listed above. For example, subsets of larger disciplines, like boxing, could potentially be broken down to "Inside/Outside Boxing" with "Inside Boxers" showing a preference for clich strikes to the head and body and "Outside Boxers" showing a preference for outside strikes to the head and body. THis approach allows for the data (i.e. the fighter's actual actions in the fight) to classify their fighting styles as opposed to their self-given label upon entry into the UFC.

Finally, because the K-means clustering algorithm is being used as a mehtod of discovery with no _ground truth labels_, we will utilize the intrinsic method of calculating and comparing the Silhouette Scores of each iteration's clustering assignments to determine the best approximation of the number of distinct fighting styles that exist in each weight class. _Proficiences_, defined as style fight averages higher than 1$$\sigma$$ from the weight class average value and  _Deficiences_, defined as style fight averages less than 1$$\sigma$$ from the weight class average, are obtained to understand the definining characteristics of each style.

#### Featherweight
![Screenshot 2025-02-26 at 5 05 29 PM](https://github.com/user-attachments/assets/82a1ce72-0fc0-48c5-9702-29f2f35d60ec)


The thres clusters accounting for 69.24% of the division's champions were styles 2,5, and 6. Of the three styles, styles 2 and 5 shared proficiences in striking, boasting hihg averages of finishes by knockout or technical knockout. Style 2 learned on the more aggresive side of stricking while Style 5 leaned on the more defensive. Style 6 was the more Brazilian Jiu-Jitsu oriented style of the groups having higher submission attempts and submission victories. Complementary to the Brazilian Jiu-Jitsu style is the groups's deficiency in takedown defense as fighters of thsi discipline are less agaisnt fighting on the ground due to more submission opportunities.

#### Lightweight
![Screenshot 2025-02-26 at 5 06 16 PM](https://github.com/user-attachments/assets/a1487f80-9157-4c58-ab50-43cfda55ba48)

The most interesting style in this division is that of style 6 which appears to be very wrestling focused with ground control time more than 3x the division average and ground strikes attempts close to 4x the division aveage. Despite the fighting style's outying nature (only having three fighters of this style), adopting a similar game appears to be highly beneficial, at least in this weight class, because despite there only being one champion in this style group, the group has yet to lose a single fight.

Another style that stands out in the lightweight division is Style 5, accounting for $$\frac{1}{4}$$ of the division's champions with 30.77% of fighters within the style group being champions. Style 5 is very similar to that of Style 2 seen in the featherweight division with prociencies in almost every striking category and having an aggressive style averaging a high number of strikes attempted per fight. However, Style 5 in the lightweight division differs from its counterpart in the featherweight division in the sense that the fighters in this division are also proficient at defense with a preference to fight at a distance. 

#### Welterweight
![Screenshot 2025-03-03 at 2 10 27 PM](https://github.com/user-attachments/assets/72dff9d6-f445-46e1-b736-f2d7b9b84a2b)

An outlier of an observation, something that is not seen in any other weight division explored here, is a style consisting of only one fighter. This one fighter, Kamaru Usman, is the division's current undefeated champion and is currently ranked as the #1 pound-for-pound figher. His averages show that he's essentially proficient at almost all striking and grappling categories, with an aversion to submissions and leg strikes. A cluster of just one fighter provides evidence that the current champion may just be on another level.

On the other hand, a more obtainable style to replicate in this division is that of Style 3. With only nine fighers of this style, three being champions, this style focuses on a volume heavy striking approach at adistance while, unlike the other styles in this division, also remainin focused on leg attacks. 

#### Middleweight
![Screenshot 2025-03-03 at 2 15 51 PM](https://github.com/user-attachments/assets/c94316fb-bc66-4570-9397-95432b3f35ce)

The middleweight division, unlike other divisions, has 5 out of the 12 styles with at least 20% of the fighres in each style being champions. However, these five styles only account for 25.16% of the fighters in the division, indicating these are pretty niche styles. Of the styles, Styles 0,2, and 9 are striking focused with very similar proficiencies. 

Style 3, another one of the five styles, hasaverage ratings across the board but shows a proficiency in the number of knockdowns they are able to score per fight. This gives an indication that at these higher weight classes simply being strong can make up for skill. Finally, style 5, teh remaining style of the five, is very wrestling heavy. Of these styles, style 2 is the most efficient with 37.50% of the fighters of this style being champions

#### Light-Heavyweight
![Screenshot 2025-03-03 at 2 19 35 PM](https://github.com/user-attachments/assets/08aecbbe-0f3a-45b4-963f-32b116701f47)

Styles 1, 2, and 5 in the light-heavyweight division make up 81.81% of the champions in the division and account for 62.70% of all the fighters in the division. Style 1 has the highest percentage of their fighters as champions at 21.43% and unlike the other weight divisions this style is submission oriented with proficiencies in only striking defense, takedown accuracy, and submission attempts.

Style 2 of this division is the only style across weight classes that is average at everything, having no proficiences or deficiencies. Such results may sign that simple well-rounded style is suitable at this weight. 

Lastly, style 5 has only three proficiencies: strikes attempted per minute. average knockdowns per fight, and wins by KO/TKO%; with 6 deficiencies in the close-quarters fighting categories (takedowns, clinch, ground). FOllowing the trend seen at middleweight, strength and knockout power appearing almost enough to win a fight.

#### Heavyweight
![Screenshot 2025-03-03 at 2 23 37 PM](https://github.com/user-attachments/assets/11ffd7fa-23d0-4f38-a6ff-ca654d6cbcd6)

Styles 1, 4, and 5 account for 66.67% of the champions within the division. Style 1 contains proficiecny in close-quarters clinch striking with fighters having the ability to reverse bad positions on the ground and prevent strikes and thuse damage. Given style 1's higher than average win by decision percentage as well, an assumption can be made that this fighter style tends to minimize risk during the fight by maintaing confrol of their opponent while landing a volume of strikes at close range. As we see power becoming more apparent at these higher weights, this style possibly shows an adaptation made by defensive minded fighters at this weight to address such threats.

Style 4, as often seen in the other weight classes, find it's success with a striking heavy style and a proficient takedown defense to boast in order to keep the fight playing to its strengths. STyle 5, is a submission-oriented style with deficiencies in many striking categories.

### Predicting Champion Model Results

The classification of a fighter in our champion prediction model will be evaluated by determinig if they actually were a champion at one point in time. A definite known fact in the equation of predicting a champion is that a winning record or a high win percentage for a fighter is a must. If we see inconsistencies with the prediction of champion percentages and the fighter's record then we will know that our approach/methodolygy needs improvements. For the final champion prediction model, some metrics that we will be evaluating on are standard classification metrics such as precision, accuracy, f1 score, confusion matrix, etc. 

__Elastic Net Model Results By Weight Class__ <br>
![Screenshot 2025-03-03 at 2 37 32 PM](https://github.com/user-attachments/assets/1c5a93db-a9a2-4e74-baf1-6fd938ce2afc)
<br>
__Random Forest Model By Weight Class__ <br>
![Screenshot 2025-03-03 at 2 38 42 PM](https://github.com/user-attachments/assets/086261b1-27f2-4130-93b0-dbef0ccd2a5c)

Upon exmaining the prediction performance of the Champion Class ("1"), the Elstic Net Regression Model seems to perform better than the Random Forsest models for most of the weight classes. The results also show that ElasticNet models tend to have trouble with precision, more than recall when comparing the two metrics side-by-side. DUe to the small data size for each weight class, along with the small champion population within each class, the metrics are a bit difficult to interpret when the confusion matrix shows the counts are so low. Most weight classes only have 2-5 champions

### Analysis of Prediction Model | Logistic Regression

Focusing on the ElasticNet model results, it's seen that certain weight classes perform better than others such as Middleweight and Heavyweight. Taking a look into the coefficients as a proxy for feature importance, we can see which features are contributing to the champion predictions best. <br>

![Screenshot 2025-03-03 at 2 45 41 PM](https://github.com/user-attachments/assets/6303bae9-870a-46b7-9a33-0d1676778dcc) <br>
![Screenshot 2025-03-03 at 2 46 13 PM](https://github.com/user-attachments/assets/9c31f0a4-35fa-4d18-ad76-af590f43163a) <br>

As these feature coefficients contribute to the prediciton of a champion, they seem to make sense as the top feature in each weight class is the "total_title_bouts" feature. Other featuers also make sense such as "max_win_streak", "wins", "avg_min_streak". Other interesting features theat tend to show up across all of the weight calsses are "Clinch_percentage", suggesting champmions or winners in general may engage in clinches more often. It can also be observed with smaller fighters, the fights are not determined by a heavy blow and could be a battle of attrition.

Focusing on the lightweight class, we can see that the cluster group fighting styles 2, 8, and 9 appear as the top features in predicting champions. Wehn comparing this with the cluster analysis, this doesn't seem to be a strong indicator of champions in comparision to other fight styles. The cluster analysis shows fighting styles 5 and 6 to be strong indicators of a champion. This mismatch may suggest that the fighting styles may not directly serve as a strong feature in prediction a champion, but that it is used as a proxy with other features

## Conclusion

The cluster analysis on fighting styles for each weight class provides useful insights on what type of fighers there are in each weight class.



