#!/usr/bin/env python
# coding: utf-8

# # PRCP-1004

# # Fifa20 Fifa 20

# # Prabhati Acharya

# The datasets provided include the players data for the Career Mode from FIFA 15 to FIFA 20 ("players_20.csv"). The data allows multiple comparison of the same players across the last 6 version of the videogame.
# 
# Some ideas of possible analysis:
# 
# - Historical comparison between Messi and Ronaldo (what skill attributes changed the most during time - compared to real-life stats);
# 
# - Ideal budget to create a competitive team (at the level of top n teams in Europe) and at which point the budget does not allow to buy significantly better players for the 11-men lineup. An extra is the same comparison with the Potential attribute for the lineup instead of the Overall attribute;
# 
# - Sample analysis of top n% players (e.g. top 5% of the player) to see if some important attributes as Agility or BallControl or Strength have been popular or not acroos the FIFA versions. An example would be seeing that the top 5% players of FIFA 20 are more fast (higher Acceleration and Agility) compared to FIFA 15. The trend of attributes is also an important indication of how some attributes are necessary for players to win games (a version with more top 5% players with high BallControl stats would indicate that the game is more focused on the technique rather than the physicial aspect).
# 
# Feel free to use the available dataset the way you prefer and do not hesitate to flag additional files (player images - datasets prior FIFA 15) that could be implemented to the existing CSV files.

# In[1]:


# Import the necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from pandas_profiling import ProfileReport
from IPython.display import Image  
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import scipy.cluster.hierarchy as shc

import warnings
warnings.simplefilter('ignore')


# In[2]:


rand_state = 10
n_iterations = 5


# In[3]:


# Load the data
fifa_data= pd.read_csv('../../Data/players_20.csv')


# In[4]:


fifa_data.head()


# ### EDA of the dataset
# 
# Perform the following exploratory data analysis on the dataset
# - Shape of the dataset
# - Info to get column names, number of not null count and data type
# - Null Count
# - Count of distinct values for each column
# - The distinct values and the counts for the categorical variables

# In[5]:


print("Shape :",fifa_data.shape)
print("Rows : ",fifa_data.shape[0])
print("Columns : ",fifa_data.shape[1])
print("\nFeatures : \n" ,fifa_data.columns.tolist())
print("\n Missing Values : ",fifa_data.isnull().sum().values.sum())
print("\nUnique Values : \n" ,fifa_data.nunique(dropna=False))


# #### Dropping columns
# 
# Next we drop the columns that we are not using for the analysis. We will drop the columns that do not add value to the analysis like the Unique ID, Player URL, Short Name, and Long Name. We drop the DOB column since it is highly correlated to the age. 
# 
# We also drop the last few columns that are concatenated with a '+' symbol because we don't have a description of the columns and don't know the purpose of these columns.

# In[6]:


fifa_reqd_data = fifa_data.loc[:, [#'sofifa_id', 'player_url', 'short_name', 'long_name', 'dob', 
                                   'age', 'height_cm', 'weight_kg', 'nationality', 
                                   'club', 'overall', 'potential', 'value_eur', 'wage_eur', 
                                   'player_positions', 'preferred_foot', 'international_reputation', 
                                   'weak_foot', 'skill_moves', 'work_rate', 'body_type', 
                                   'real_face', 'release_clause_eur', 'player_tags', 'team_position', 
                                   'team_jersey_number', 'loaned_from', 'joined', 'contract_valid_until', 
                                   'nation_position', 'nation_jersey_number', 'pace', 'shooting', 'passing', 
                                   'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling', 
                                   'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning', 'player_traits', 
                                   'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 
                                   'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 
                                   'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 
                                   'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 
                                   'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 
                                   'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 
                                   'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 
                                   'mentality_penalties', 'mentality_composure', 'defending_marking', 
                                   'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 
                                   'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 
                                   'goalkeeping_reflexes'#, 
                                   #'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 
                                   #'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 
                                   #'cb', 'rcb', 'rb'
                                  ]
                              ]


# In[7]:


fifa_reqd_data


# ###### Grouping the Data into Categorical and Continuous Variable

# In[8]:


#Grouping the Data into Categorical and Continuous Variable
""" 
'player_url', 'short_name', 'long_name', 'dob', , 'ls', 'st', 'rs', 'lw', 'lf', 'cf',
             'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb',
             'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb'
""" 
Catvar_list=['nationality', 'club', 'player_positions', 'preferred_foot', 'work_rate', 'body_type',
             'real_face', 'player_tags', 'team_position', 'loaned_from', 'joined',
             'nation_position', 'player_traits']

# 'sofifa_id', 
Convar_list=['age', 'height_cm', 'weight_kg', 'overall', 'potential',
             'value_eur', 'wage_eur', 'international_reputation', 'weak_foot',
             'skill_moves', 'release_clause_eur', 'team_jersey_number',
             'contract_valid_until', 'nation_jersey_number', 'pace', 'shooting',
             'passing', 'dribbling', 'defending', 'physic', 'gk_diving',
             'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed',
             'gk_positioning', 'attacking_crossing', 'attacking_finishing',
             'attacking_heading_accuracy', 'attacking_short_passing',
             'attacking_volleys', 'skill_dribbling', 'skill_curve',
             'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
             'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
             'movement_reactions', 'movement_balance', 'power_shot_power',
             'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
             'mentality_aggression', 'mentality_interceptions',
             'mentality_positioning', 'mentality_vision', 'mentality_penalties',
             'mentality_composure', 'defending_marking', 'defending_standing_tackle',
             'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 
             'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']


# #### Checking the unique values in the Categorical columns
# 
# We will run the categorical columns in a loop and get the unique values of these columns. This will help us examine these columns manually and give us a general idea about the dataset. 

# In[9]:


for cat in Catvar_list:
    print('\nUnique values of Project_Data.'+cat+':: \n',fifa_reqd_data[cat].value_counts(dropna = False))


# #### Checking for count of NaN values
# 
# Since we had a huge number of columns, the output was not efficient when we check the number of NaN values for all columns. So, we modified it a bit and are displaying only the columns that have at least 1 NaN value in our output. 
# 
# We also output the NaN into a heatmap for easy visualization.
# 
# Then we drop the columns that have a high NaN percentage. We also impute the Goal keeper related fields with 0, the team position, jersey number columns with 0, and the other columns with Median.

# In[10]:


for column in fifa_reqd_data.columns[fifa_reqd_data.isnull().any()]:
    print(column,"\t",fifa_reqd_data[column].isnull().sum())


# In[11]:


sns.set_style('whitegrid')
sns.heatmap(fifa_reqd_data.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[12]:


fifa_reqd_data_new = fifa_reqd_data.drop(columns=['player_tags', 'loaned_from', 'nation_position'
                                                  , 'nation_jersey_number', 'gk_diving', 'gk_handling'
                                                  , 'gk_kicking', 'gk_reflexes', 'gk_speed'
                                                  , 'gk_positioning', 'player_traits', 'joined'], axis=1)


# In[13]:


# Following are missing for Goalkeepers. Defaulting to 0
# pace, shooting, passing, dribbling, defending, physic

columns_missing_gk = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']

fifa_reqd_data_new[columns_missing_gk] = fifa_reqd_data_new[columns_missing_gk].fillna(0)
    
columns_missing = ['team_position', 'team_jersey_number', 'contract_valid_until']

fifa_reqd_data_new[columns_missing] = fifa_reqd_data_new[columns_missing].fillna(0)

columns_release = ['release_clause_eur']

fifa_reqd_data_new[columns_release] = fifa_reqd_data_new[columns_release].fillna(fifa_reqd_data_new[columns_release].median())


# In[14]:


sns.set_style('whitegrid')
sns.heatmap(fifa_reqd_data_new.isnull(),yticklabels=False,cbar=True,cmap='viridis')


# In[15]:


for column in fifa_reqd_data_new.columns[fifa_reqd_data_new.isnull().any()]:
    print(column,"\t",fifa_reqd_data_new[column].isnull().sum())


# ### Encoding Categorical Variables
# 
# We will proceed to encode all the variables with __object__ data type. We will also store all the mapping between the Feature values and the corresponding encoded values in a dataframe so that we can refer back to it in order to explain the interactions between the features. 

# In[16]:


labelencoder = LabelEncoder()


# In[17]:


fifa_reqd_data_enc = fifa_reqd_data_new


# In[18]:


objList = fifa_reqd_data_enc.select_dtypes(include = "object").columns
objList


# In[19]:


Feature_Code_Value = pd.DataFrame(columns=['Feature_Type', 'Feature_Value', 'Feature_Value_Enc', 'Feature_Value_Cnt'])


# In[20]:


for feat in objList:
    feat_col_name = feat+'_enc'
    fifa_reqd_data_enc[feat_col_name] = labelencoder.fit_transform(fifa_reqd_data_enc[feat].astype(str))
    feat_temp_val = fifa_reqd_data_enc.groupby([feat, feat_col_name]).size().reset_index().rename(columns={0:'Feature_Value_Cnt'})
    feat_temp_val['Feature_Type'] = feat
    #feat_temp_val[['Feature_Type', feat, feat_col_name, 'Feature_Value_Cnt']]
    Feature_Code_Value = Feature_Code_Value[['Feature_Type', 'Feature_Value', 'Feature_Value_Enc', 'Feature_Value_Cnt']].append(feat_temp_val[['Feature_Type', feat, feat_col_name, 'Feature_Value_Cnt']])
    Feature_Code_Value.Feature_Value.fillna(Feature_Code_Value[feat], inplace=True)
    Feature_Code_Value.Feature_Value_Enc.fillna(Feature_Code_Value[feat_col_name], inplace=True)
    Feature_Code_Value.drop(columns=[feat, feat_col_name], axis=1, inplace=True)
    del feat_temp_val
    fifa_reqd_data_enc.drop(columns=[feat], axis=1, inplace=True)
    
Feature_Code_Value.reset_index(drop=True, inplace=True)
Feature_Code_Value['Feature_Value_Enc'] = Feature_Code_Value['Feature_Value_Enc'].astype(int)


# In[21]:


Feature_Code_Value


# ## Scaling the data set
# 
# It is a step of Data Pre Processing which is applied to independent variables or features of data. It basically helps to normalise the data within a particular range. Sometimes, it also helps in speeding up the calculations in an algorithm. The Features have been scaled to a mean of 0 and variance of 1 to improve accuracy of the classification models.
# 
# *fit_transform* within MinMaxScaler() function fits to data, then transform it. Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.

# In[22]:


scaler = MinMaxScaler()


# In[23]:


fifa_reqd_data_enc_scaled = pd.DataFrame(scaler.fit_transform(fifa_reqd_data_enc), columns=fifa_reqd_data_enc.columns)


# In[24]:


corr = fifa_reqd_data_enc_scaled.corr()

kot = corr[(corr>=.8) & (corr<1)]
kot.dropna(axis=0, how='all', inplace = True)
kot.dropna(axis=1, how='all', inplace = True)
plt.figure(figsize=(25,25))
#sns.heatmap(kot, annot=True, fmt='.4g', cmap= 'coolwarm', linewidths=3, linecolor='black')

matrix = np.triu(kot)
sns.heatmap(kot, annot=True, fmt='.4g', cmap= 'coolwarm', mask = matrix)


# In[25]:


kot.to_csv('Correlation.csv')


# #### Droping highly correlated columns
# 
# We then identify the highly correlated (> 0.8) columns and drop them. 

# In[26]:


corr_matrix = fifa_reqd_data_enc_scaled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

to_drop


# In[27]:


fifa_reqd_data_no_corr = fifa_reqd_data_enc_scaled.drop(columns=to_drop, axis=1)


# ###### Profiling the dataset
# 
# Now that we have a good grasp of the columns that we want to profile on, we will proceed with the data profiling step. 
# 
# Generates profile reports from a pandas DataFrame. The pandas df.describe() function is great but a little basic for serious exploratory data analysis. pandas_profiling extends the pandas DataFrame with df.profile_report() for quick data analysis.
# 
# For each column the following statistics - if relevant for the column type - are presented in an interactive HTML report:
# 
# - Type inference: detect the types of columns in a dataframe.
# - Essentials: type, unique values, missing values
# - Quantile statistics like minimum value, Q1, median, Q3, maximum, range, interquartile range
# - Descriptive statistics like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
# - Most frequent values
# - Histograms
# - Correlations highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices
# - Missing values matrix, count, heatmap and dendrogram of missing values
# - Duplicate rows Lists the most occurring duplicate rows
# - Text analysis learn about categories (Uppercase, Space), scripts (Latin, Cyrillic) and blocks (ASCII) of text data

# In[28]:


ProfileReport(fifa_reqd_data_no_corr)


# In[29]:


#prof = ProfileReport(fifa_reqd_data_no_corr)
#prof.to_file(output_file='output.html')


# ## K Means Clustering
# 
# The k-means clustering method is an unsupervised machine learning technique used to identify clusters of data objects in a dataset. There are many different types of clustering methods, but k-means is one of the oldest and most approachable. These traits make implementing k-means clustering in Python reasonably straightforward, even for novice programmers and data scientists.
# 
# Clustering is a set of techniques used to partition data into groups, or clusters. Clusters are loosely defined as groups of data objects that are more similar to other objects in their cluster than they are to data objects in other clusters. In practice, clustering helps identify two qualities of data:
# 
# - Meaningfulness
# - Usefulness
# 
# __Meaningful__ clusters expand domain knowledge. For example, in the medical field, researchers applied clustering to gene expression experiments. The clustering results identified groups of patients who respond differently to medical treatments.
# 
# __Useful__ clusters, on the other hand, serve as an intermediate step in a data pipeline. For example, businesses use clustering for customer segmentation. The clustering results segment customers into groups with similar purchase histories, which businesses can then use to create targeted advertising campaigns.
# 
# Conventional k-means requires only a few steps. The first step is to randomly select k centroids, where k is equal to the number of clusters you choose. Centroids are data points representing the center of a cluster.
# 
# The main element of the algorithm works by a two-step process called expectation-maximization. The expectation step assigns each data point to its nearest centroid. Then, the maximization step computes the mean of all the points for each cluster and sets the new centroid.

# In[30]:


fifa_reqd_data_no_corr.columns


# In[31]:


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(fifa_reqd_data_no_corr)


# In[32]:


# inertia on the fitted data
kmeans.inertia_


# In[33]:


# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(fifa_reqd_data_no_corr)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[34]:


# k means using 6 clusters and k-means++ initialization
kmeans = KMeans(n_jobs = -1, n_clusters = 6, init='k-means++')
kmeans.fit(fifa_reqd_data_no_corr)
pred = kmeans.predict(fifa_reqd_data_no_corr)


# In[35]:


frame = pd.DataFrame(fifa_reqd_data_no_corr)
frame['cluster'] = pred
frame['cluster'].value_counts()


# In[36]:


fifa_reqd_data_no_corr['cluster'] = pred
fifa_data['cluster'] = pred


# In[37]:


fifa_data


# ### K-Means using PCA

# In[38]:


preprocessor = Pipeline(
    [
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)


# In[39]:


clusterer = Pipeline(
    [
        (
            "kmeans",
            KMeans(
                n_clusters=6,
                init="k-means++",
                n_init=50,
                max_iter=500,
                random_state=42,
            ),
        ),
    ]
)


# In[40]:


pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)


# In[41]:


pipe.fit(fifa_reqd_data_no_corr)


# In[42]:


preprocessed_data = pipe["preprocessor"].transform(fifa_reqd_data_no_corr)
predicted_labels = pipe["clusterer"]["kmeans"].labels_
silhouette_score(preprocessed_data, predicted_labels)


# In[43]:


pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(fifa_reqd_data_no_corr),
    columns=["component_1", "component_2"],
)

pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
#pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))

scat = sns.scatterplot(
    "component_1",
    "component_2",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    #style="true_label",
    palette="Set2",
)

scat.set_title(
    "Clustering results from FIFA20 Data"
)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()


# ### Hierarchical Clustering 
# 
# There are two types of hierarchical clustering: Agglomerative and Divisive. In the former, data points are clustered using a bottom-up approach starting with individual data points, while in the latter top-down approach is followed where all the data points are treated as one big cluster and the clustering process involves dividing the one big cluster into several small clusters.
# 
# Once one large cluster is formed by the combination of small clusters, dendrograms of the cluster are used to actually split the cluster into multiple clusters of related data points.
# 
# The algorithm starts by finding the two points that are closest to each other on the basis of Euclidean distance. The vertical height of the dendogram shows the Euclidean distances between points. The next step is to join the cluster formed by joining two points to the next nearest cluster or point which in turn results in another cluster. This process continues until all the points are joined together to form one big cluster.
# 
# Once one big cluster is formed, the longest vertical distance without any horizontal line passing through it is selected and a horizontal line is drawn through it. The number of vertical lines this newly created horizontal line passes is equal to number of clusters.

# In[44]:


fifa_reqd_data_enc_hie = fifa_reqd_data_enc[fifa_reqd_data_enc['overall'] > 85]


# In[45]:


fifa_reqd_data_enc_hie_scaled = pd.DataFrame(scaler.fit_transform(fifa_reqd_data_enc_hie), columns=fifa_reqd_data_enc_hie.columns)


# In[46]:


kot.to_csv('Correlation.csv')


# #### Droping highly correlated columns
# 
# We then identify the highly correlated (> 0.8) columns and drop them. 

# In[47]:


corr_matrix = fifa_reqd_data_enc_hie_scaled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

to_drop


# In[48]:


fifa_reqd_data_hie_no_corr = fifa_reqd_data_enc_hie_scaled.drop(columns=to_drop, axis=1)


# In[49]:


names = fifa_data[fifa_data['overall'] > 85]['short_name'].tolist()


# In[50]:


plt.figure(figsize=(25,25))
plt.title('Hierarchical Clustering Dendrogram with Average Linkage')
dendrogram = shc.dendrogram(shc.linkage(fifa_reqd_data_hie_no_corr, method="average"), 
                            labels= names, leaf_font_size = 13, orientation='right')


# In the above Dendrogram, Messi is in a cluster of his own. However, his cluster is closest to that of M. Å kriniar who is on the other side of the spectrum. On looking closer, Neymar Jr., Ronaldo, and Saurez are all in clusters farther away from Messi. 
# 
# We'll look at clustering them using a different method. 

# In[51]:


plt.figure(figsize=(25,25))
plt.title('Hierarchical Clustering Dendrogram with Complete Linkage')
dendrogram = shc.dendrogram(shc.linkage(fifa_reqd_data_hie_no_corr, method="complete"), 
                            labels= names, leaf_font_size = 13, orientation='right')


# In the second Dendrogram, the representation is much better. If we draw a line through the point where we have 6 clusters, here too, Messi is in a cluster of his own. This shows his dominance over the rest of the field. This dendrogram is better than the first one because Messi is closer to other great players like Neymar, Ronaldo, Salah, and Saurez. 

# # Conclusion
# 
# We started by importing the necessary packages and looking at unique values in the categorical columns. Since the data set is quite big and has a lot of features, we are not performing the pandas_profiling immediately. We will manually complete feature selection before we proceed with in-depth profiling. 
# 
# We also performed data imputation, data encoding, and scaling on the dataset. Finally, we drop highly correlated columns by looking at the correlation matrix. Once we have completed the data preprocessing steps, we move on the complete the detailed profiing using __pandas_profiling__. 
# 
# Next we move on to clustering algorithms. We start with the full dataset and run K-Means clustring algorithm in a loop to identify the ideal number of clusters. In our case, we chose to proceed with 6 clusters. We also created an additional column in our dataset to indicate the cluster that the player would belong to. We also applied dimensionality reduction using PCA on the dataset to use only 2 components. We visualized the clusters after applying dimensionality reduction. 
# 
# Finally, we moved on to Hierarchical clustering. The advantage with hierarchical clustering is that it is great for visually understanding the constituents of the cluster. Since we had a big dataset and we wanted to render the dendrogram neatly, we took a subset of our data (Overall > 85). The corresponding rendering of the dataset clearly shows us that our choice of 6 clusters is a good one.
