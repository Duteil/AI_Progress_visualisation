# -*- coding: utf-8 -*-
"""
Visualisation of the relative impact of papers published in the Vision, 
Language, and Games domain.
The data come from https://epochai.org/mlinputs/visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression

#%% Analysis of Games, Language and Vision

path_csv = "Parameter, Compute and Data Trends in Machine Learning - ALL ML SYSTEMS.csv"
data = pd.read_csv(path_csv)
# Some papers are mentioned several times, so duplicates should be ignored
data = data.drop_duplicates(subset=["Reference"], keep=False)

df1 = data[['Domain', 'Year', 'Citations']]

# We are interested in Vision, Language and Games.
print(set(df1['Domain'].values))
df1['Domain'] = df1['Domain'].replace(['VIsion'], 'Vision')
df = df1[df1['Domain'].isin(['Vision', 'Language', 'Games'])]
df = df.dropna()

# To consider the relative importance of each field, I consider that the 
# relevant variable is the total number of citations per year for a given field
graph_df = df.pivot_table(index='Year', columns='Domain', values='Citations', aggfunc='sum')
graph_df = graph_df.fillna(0)
graph_df.plot.area(ylabel='Total number of citations')


#%% Analysis of a different trend


# For the next question, I count the number of NaN per column to determine which 
# values have enough data to make a model out of them.
non_nan_counts = data.count()

# We are now focusing on the name of the organization, as well as its categorization.
# First, some formatting issues must be solved. The following lines ensure all 
# organizations are separated by a comma.
newdata = data[['Organization(s)', 'Organization Categorization', 'Year']]
newdata.loc[newdata['Organization(s)'] == "Mitsubishi Electric Research Labs and Compaq CRL", 'Organization(s)'] = "Mitsubishi Electric Research Labs, Compaq CRL"
newdata.loc[newdata['Organization(s)'] == 'Cambridge University Engineering & Carnegie Mellon University', 'Organization(s)'] = 'Cambridge University Engineering, Carnegie Mellon University'
newdata.loc[newdata['Organization(s)'] == 'Carnegie Mellon University & ATR Interpreting Telephony Research Laboratories & University of Toronto ', 'Organization(s)'] = 'Carnegie Mellon University, ATR Interpreting Telephony Research Laboratories, University of Toronto'
newdata.loc[newdata['Organization(s)'] == 'Chinese Academy of Sciences ; University of Oxford', 'Organization(s)'] = 'Chinese Academy of Sciences, University of Oxford'
newdata.loc[newdata['Organization(s)'] == 'IDSIA and TU Munich', 'Organization(s)'] = 'IDSIA, TU Munich'
newdata.loc[newdata['Organization(s)'] == 'MIT and Stanford', 'Organization(s)'] = 'MIT, Stanford'
newdata.loc[newdata['Organization(s)'] == 'RWTH Aachen and University of Southern California', 'Organization(s)'] = 'RWTH Aachen, University of Southern California'
newdata.loc[newdata['Organization(s)'] == 'Stanford and UC Berkeley', 'Organization(s)'] = 'Stanford, UC Berkeley'
newdata.loc[newdata['Organization(s)'] == 'Technische Universität Wien Austria & University of California', 'Organization(s)'] = 'Technische Universität Wien Austria, University of California'
newdata.loc[newdata['Organization(s)'] == 'University of California San Diego & Shannon Laboratory', 'Organization(s)'] = 'University of California San Diego, Shannon Laboratory'
newdata.loc[newdata['Organization(s)'] == 'University of California and University of Carnegie Mellon', 'Organization(s)'] = 'University of California, University of Carnegie Mellon'
newdata.loc[newdata['Organization(s)'] == 'University of Chicago & Toyota Technological Institute', 'Organization(s)'] = 'University of Chicago, Toyota Technological Institute'
newdata.loc[newdata['Organization(s)'] == 'University of Colorado & New Mexico State University', 'Organization(s)'] = 'University of Colorado, New Mexico State University'
newdata.loc[newdata['Organization(s)'] == 'University of the Balearic Islands and CMLA', 'Organization(s)'] = 'University of the Balearic Islands, CMLA'

contributors = newdata.assign(Organization=newdata['Organization(s)'].str.split(',')).explode('Organization')
contributors = contributors.drop('Organization(s)', axis=1)
contributors.reset_index(inplace = True)
contributors.at[149,'Organization Categorization'] = 'Academia'
contributors.at[150,'Organization Categorization'] = 'Industry'
contributors = contributors.drop(columns=['index'])
contributors = contributors.dropna()
organizations = set(contributors['Organization'].values)

# Next, I create a dictionary to gather all similar groups under the same name.
name_map = {key: None for key in organizations}

for key in name_map:
    if 'Microsoft' in key:
        name_map[key] = 'Microsoft'
    elif 'Google' in key or 'DeepMind' in key or 'Deepmind' in key:
        name_map[key] = 'Google'
    elif 'IBM' in key:
        name_map[key] = 'IBM'
    elif 'Facebook' in key or 'Meta' in key:
        name_map[key] = 'Meta'
    elif 'AT&T' in key:
        name_map[key] = 'AT&T'
    elif 'Baidu' in key:
        name_map[key] = 'Baidu'
    elif 'Alibaba' in key:
        name_map[key] = 'Alibaba'
    elif 'NVIDIA' in key or 'Nvidia' in key:
        name_map[key] = 'Nvidia'
    elif 'Xerox' in key :
        name_map[key] = 'Xerox'
    elif 'NHK' in key :
        name_map[key] = 'NHK'

remaining_keys = set(name_map.keys())
for ele in name_map:
    if name_map[ele] is not None:
        remaining_keys.remove(ele)
        
other_values = {
                ' Aachen University': 'RWTH',
                ' Allen Institute for AI': 'AI2',
                ' Apple': 'Apple',
                ' ATR Interpreting Telephony Research Laboratories': 'ATR',
                ' Australian Centre for Robotic Vision': 'Australian Centre for Robotic Vision',
                ' Australian National University': 'Australian National University',
                ' BAAI': 'BAAI',
                ' Beihang University': 'Beihang University',
                ' Berkeley': 'Berkeley',
                ' BigScience': 'BigScience',
                ' Brain Team': 'Google',
                ' Brain team': 'Google',
                ' Brown University': 'Brown',
                ' CENPARMI': 'CENPARMI',
                ' CIFAR': 'CIFAR',
                ' CMU': 'CMU',
                ' CalTech': 'Caltech',
                ' Canadian Institute for Advanced Research and Vector Institute': 'Vector Institute',
                ' Carnegie Mellon University': 'Carnegie Mellon',
                ' Charles University': 'Charles University',
                ' Chicago & University of California': 'UC Irvine',
                ' Chinese Academy of Sciences': 'Chinese Academy of Sciences',
                ' Chinese University of Hong Kong': 'Chinese University of Hong Kong',
                ' CMLA': 'Cachan',
                ' College Park': 'University of Maryland',
                ' Columbia': 'Columbia',
                ' Compaq CRL': 'Compaq',
                ' Cornell': 'Cornell',
                ' Courant Institute of Mathematical Sciences': 'Courant',
                ' Czech Technical University': 'Czech Technical University',
                ' Ecole': 'ENS',
                ' Ecole Normale': 'ENS',
                ' HEC': 'HEC',
                ' HERE Technologies': 'HERE',
                ' Harvard University': 'Harvard',
                ' Hong Kong Polytechnic University': 'Hong Kong Polytechnic University',
                ' IARIA Vienna': 'IARIA',
                ' IDIAP': 'IDIAP',
                ' IDSIA': 'IDSIA',
                ' INRIA': 'INRIA',
                ' INRIA Grenoble': 'INRIA',
                ' Inria': 'INRIA',
                ' Insight Centre NUI Galway': 'Insight',
                ' Intel Labs': 'Intel',
                ' Inteligent Systems Lab Amsterdam': 'ISLA',
                ' Irvine': 'UC Irvine',
                ' Jacobs University': 'Jacobs',
                ' Jacobs University Bremen': 'Jacobs',
                ' Jagiellonian University':'Jagiellonian',
                ' Johannes Kepler University': 'JKU',
                ' Johns Hopkins University': 'Johns Hopkins',
                ' LEAR Team': 'INRIA',
                ' MIT': 'MIT',
                ' Med AI Technology': 'Med AI',
                ' Megvii Inc': 'Megvii',
                ' Mila- Quebec AI': 'Mila',
                ' New Mexico State University': 'NMSU',
                ' NUANCE Communications': 'NUANCE',
                ' NUS': 'NUS',
                ' NYU': 'NYU',
                ' Nanyang Technological University': 'Nanyang',
                ' National Institute of Informatics': 'NII',
                ' Oak Ridge National Laboratory': 'Oak Ridge',
                ' OpenAI': 'OpenAI',
                ' Peking University': 'Peking',
                ' Princeton': 'Princeton',
                ' Ritsumeikan University': 'Ritsumeikan',
                ' Runway': 'Runway',
                ' Rutgers University': 'Rutgers',
                ' San Diego': 'UCSD',
                ' Santa Cruz': 'UCSC',
                ' Shandong University': 'Shandong',
                ' Shanghai Qi Zhi institute': 'Qi Zhi',
                ' Shannon Laboratory': 'Shannon Laboratory',
                ' Shenzhen Institute of Advanced Technology': 'SIAT',
                ' Stanford': 'Stanford',
                ' Stanford University': 'Stanford',
                ' Swiss Federal Institute of Technology ': 'ETH',
                ' TTIC': 'TTIC',
                ' TU Munich': 'TUM',
                ' TUM': 'TUM',
                ' Technion- Israel Institute of Technology': 'Technion',
                ' Texas A&M': 'Texas A&M',
                ' The Chinese University of Hong Kong': 'CUHK',
                ' Toyota Technological Institute at Chicago': 'TTIC',
                ' Tsinghua university': 'Tsinghua',
                ' Twitter': 'Twitter',
                ' UC Berkeley': 'Berkeley',
                ' UIUC': 'UIUC',
                ' ULSee Inc.': 'ULSee',
                ' UMass Lowell': 'Lowell',
                ' University College London': 'UCL',
                ' University King College': 'KCL',
                ' University de Montreal': 'Montreal',
                ' University du Maine': 'Maine',
                ' University of Adelaide': 'Adelaide',
                ' University of Amsterdam': 'Amsterdam',
                ' University of California Los Angeles': 'UCLA',
                ' University of Illinois at Urbana- Champaigne': 'UIUC',
                ' University of Michigan': 'Michigan',
                ' University of Montreal': 'Montreal',
                ' University of North Carolina': 'UNC',
                ' University of Oxford': 'Oxford',
                ' University of Pittsburgh': 'Pittsburgh',
                ' University of Rochester': 'Rochester',
                ' University of Sherbrooke': 'Sherbrooke',
                ' University of Southern California': 'USC',
                ' University of Technology Sydney': 'UTS',
                ' University of Texas': 'UT Austin',
                ' University of Texas at San Antonio': 'UTSA',
                ' University of Toronto': 'Toronto',
                ' University of Washington': 'UW',
                ' Xi’an Jiaotong University': 'XJTU',
                ' Yahoo Research': 'Yahoo',
                ' Yonsei University': 'Yonsei',
                'AI2': 'AI2',
                'AI21 Labs': 'AI21',
                'AI21labs': 'AI21',
                'ATR Labs': 'ATR',
                'Aalborg University': 'Aalborg',
                'Air Force Institute of Technology': 'AFIT',
                'AllenAI': 'AI2',
                'Amazon': 'Amazon',
                'American University of Beirut': 'AUB',
                'BAAI': 'BAAI',
                'Bell Laboratories': 'Bell',
                'Berkeley': 'Berkeley',
                'Biological Cybernetics': 'Biological Cybernetics',
                'Brno University of Technology': 'VUT',
                'Brown University': 'Brown',
                'CNRS': 'CNRS',
                'CRAN': 'CRAN',
                'California Institute of Technology': 'CIT',
                'Cambridge University Engineering': 'Cambridge',
                'Carnegie Mellon University': 'Carnegie Mellon',
                'Carnegie Mellon University ': 'Carnegie Mellon',
                'Chinese Academy of Sciences': 'CAS',
                'Chinese University of Hong Kong': 'CUHK',
                'Collège de France': 'Collège de France',
                'Cornell Aeronautical Laboratory': 'Cornell',
                'DeepScale': 'DeepScale',
                'ETH Zurich': 'ETH',
                'EURECOM': 'EURECOM',
                'EleutherAI': 'EleutherAI',
                'Ellis Unit Linz and LIT AI Lab': 'JKU',
                'Graz University of Technology': 'TU Graz',
                'Harbin Institute of Technology': 'HITSZ',
                'Harvard': 'Harvard',
                'Harvard University Psychological Laboratories': 'Harvard',
                'Heidelberg University': 'Heidelberg',
                'Helsinki University of Technology': 'TKK',
                'Heriot-Watt University': 'Heriot-Watt',
                'Hugging Face': 'Hugging Face',
                'HuggingFace': 'Hugging Face',
                'IDSIA': 'IDSIA',
                'IDSIA ; University of Lugano & SUPSI': 'IDSIA',
                'IDSIA Switzerland': 'IDSIA',
                'IEEE': 'Stanford',
                'INRIA': 'INRIA',
                'Indian Statistical Institute': 'ISICAL',
                'Inria Grenoble Rhône-Alpes': 'INRIA',
                'Inspur': 'Inspur',
                'Institute for Advanced Study': 'Vanderbilt',
                'Institute of Advanced Research in Artificial Intelligence': 'IARAI',
                'Instituto de Ciencias Aplicadas y Technologia': 'ICAT',
                'Johannes Kepler University Linz': 'JKU',
                'Johns Hopkins University': 'Johns Hopkins',
                'Karlsruhe Institute of Technology': 'KIT',
                'Korea Advanced Institute of Science and Technology': 'KAIST',
                'MIT': 'MIT',
                'Machine Vision group': 'MVG',
                'Massachusetts Institute of Technology': 'MIT',
                'Massachusetts Institute of Technology (MIT)': 'MIT',
                'Megvii Inc': 'Megvii',
                'Mitsubishi Electric Research Labs': 'Mitsubishi',
                'Montreal Institute for learning Algorithms': 'Mila',
                'NAVER AI Lab': 'Naver',
                'NEC Laboratories': 'Nec',
                'NTT Communication Science Laboratories': 'NTT Communication Science Laboratories',
                'NUS': 'NUS',
                'NYU': 'NYU',
                'Nanjing University': 'Nanjing',
                'National Chiao Tung University': 'NYCU',
                'National University of Singapore': 'NUS',
                'Naver Corp': 'Naver',
                'Netflix': 'Netflix',
                'New York University': 'NYU',
                'Northeastern University': 'Northeastern',
                'Open AI': 'OpenAI',
                'OpenAI': 'OpenAI',
                'PanGu-α team': 'Huawei',
                'Pragmatic Theory Inc.': 'Pragmatic Theory',
                'Preferred Networks Inc': 'Preferred Networks',
                'Princeton University': 'Princeton',
                'RWTH Aachen - University of Technology': 'RWTH',
                'RWTH Aachen': 'RWTH',
                'Roke Manor Research': 'Roke',
                'SVCL UC San Diego': 'UCSD',
                'Salesforce': 'Salesforce',
                'Salesforce research': 'Salesforce',
                'Sandia Corporation': 'Sendia',
                'Seoul National University': 'SNU',
                'Soongsil University': 'Soongsil',
                'Stability AI': 'Stability AI',
                'Stanford': 'Stanford',
                'Stanford Research Institute': 'Stanford',
                'Stanford University': 'Stanford',
                'TU Darmstadt': 'TU Darmstadt',
                'Technical University of Munich': 'TUM',
                'Technische Universität Wien Austria': 'TUW',
                'Tel Aviv University': 'TAU',
                'The Chinese University of Hong Kong': 'CUHK',
                'The Robotics Institute': 'Carnegie Mellon',
                'The Technical University of Munich': 'TUM',
                'The University of Genoa': 'Genoa',
                'Tsinghua KEG': 'BAAI',
                'Tsinghua University': 'BAAI',
                'Twitter': 'Twitter',
                'UC Berkeley': 'Berkeley',
                'UC Davis': 'UC Davis',
                'UC San Diego': 'UCSD',
                'UT Austin': 'UT Austin',
                'Uber AI': 'Uber',
                'Univeristy of Amsterdam': 'Amsterdam',
                'Univeristy of California Berkley': 'Berkeley',
                'Univeristy of Lubeck': 'Lubeck',
                'Univeristy of Toronto': 'Toronto',
                'Universidad Nacional de Cordoba': 'Cordoba',
                'Universite de Montréal': 'Montreal',
                'University College London': 'UCL',
                'University of Adelaide': 'Adelaide',
                'University of Alberta': 'Alberta',
                'University of Amsterdam': 'Amsterdam',
                'University of Bern': 'Bern',
                'University of California': 'UCSD',
                'University of Cambridge': 'Cambridge',
                'University of Canterbury': 'Canterbury',
                'University of Chicago': 'Chicago',
                'University of Colorado': 'Colorado',
                'University of Edinburgh': 'Edinburgh',
                'University of Erlangen - Nuremburg': 'FAU',
                'University of Essex': 'Essex',
                'University of Freiburg': 'Freiburg',
                'University of Geneva': 'Geneva',
                'University of Guelph': 'Guelph',
                'University of Illinois': 'Illinois',
                'University of London': 'London',
                'University of Maryland': 'Maryland',
                'University of Michigan': 'Michigan',
                'University of Minnesota': 'Minnesota',
                'University of Montreal': 'Montreal',
                'University of Oslo': 'Oslo',
                'University of Oxford': 'Oxford',
                'University of Pennsylvania': 'Pennsylvania',
                'University of Rochester': 'Rochester',
                'University of San Francisco': 'USFCA',
                'University of Southern California': 'USC',
                'University of Stanford': 'Stanford',
                'University of Sussex': 'Sussex',
                'University of Technology Sydney': 'UTS',
                'University of Texas': 'UT Austin',
                'University of Toronto': 'Toronto',
                'University of Washington': 'UW',
                'University of Wisconsin Madison': 'WISC',
                'University of the Balearic Islands': 'UIB',
                'Université Paris-Est': 'Ponts',
                'Université de Montréal': 'Montreal',
                'Utrecht University': 'Utrecht',
                'Visual Computing Institute': 'RWTH',
                'Warsaw University': 'Warsaw',
                'Xiamen University': 'Xiamen',
                'École des Ponts ParisTech': 'Ponts'}        
        
for key in other_values:
    name_map[key] = other_values[key]
for key, value in dict(name_map).items():
    if value is None:
        del name_map[key]

# All organizations are replaced by their new name.
contributors['Organization'] = contributors['Organization'].map(name_map)
# The nan are the publications that have no associated organization. This can 
# happen if the name of an organization included a comma for instance:
contributors = contributors.dropna()
contributors = contributors.drop('Organization Categorization', axis=1)                

# Number of organizations having published impactful papers by year:
year_counts = contributors.groupby('Year')['Organization'].nunique()
year_counts.plot()

# I am excluding 2022 as the most influential papers might be missing from the dataset.
# Moreover, it seems the number of impactful papers per year has been about constant 
# until 2006, after which it started increasing, so I am considering the data from that point.
past_years = year_counts[17:-1]
                
X = np.array(past_years.index)
y = np.array(past_years.values)

# Creating a LinearRegression model:
model = LinearRegression()

# Fiting the model to the data:
model.fit(X, y)

# Model parameters:
intercept = model.intercept_
coefficient = model.coef_[0]

# Predictions for future years
print("Intercept:", intercept)
print("Coefficient:", coefficient)

# Use the model to make predictions for future years
time_range = range(2006,2026)
X_future = np.array(time_range).reshape(-1, 1)
predictions = model.predict(X_future)

print(predictions)

# Displaying the trend and the predictions:
fig, ax = plt.subplots()
ax.plot(X, y, color='blue')
ax.plot(time_range, predictions, color='red')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Organizations')
ax.set_title('Number of Organizations Established Over Time')
plt.xlim([1950, 2025])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()