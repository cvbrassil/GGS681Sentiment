# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:09:22 2021

@author: connor.brassil
"""
##############################INITIAL SETUP###################################
####Import required libraries

import pandas as pd
import json
import os
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
from textblob import TextBlob
from math import radians, degrees, cos, sin, sqrt, atan2
#from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import random
import hdbscan
import cartopy
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
import matplotlib.patches as mpatches
import statsmodels.api as sm
from patsy import dmatrices

####Set directory, filenames that store unprocessed JSON tweets
os.chdir('directory')

filelist = ['comma separated file list']


####Declare Twitter object vars of interest and create a blank pandas dataframe

twtvars = ['id_str','user','created_at','lang','text',
           'source','coordinates','place','extended_tweet']

atdf = pd.DataFrame(columns = twtvars)


####Iterate our files format as JSON, add to local df, append to master df

for i in filelist:
    thisdf = pd.DataFrame()
    tweetdata = []
    with open(i,"r") as tweetfile: 
        print("Reading File: " + i)
        for line in tweetfile:
            tweet = json.loads(line)
            tweetdata.append(tweet)
    thisdf = pd.DataFrame(tweetdata,columns = twtvars)
    atdf = pd.concat([atdf,thisdf], sort=False)
         
print("File reading complete, atdf data frame created")

print(atdf.info())
    
#############################DATA PROCESSING###################################    
####Data processing  
#Rename initial coordinates field
#Convert datetime
#Regular expression to get only what's between HTML tags: > <
#Drop if no place
#Drop duplicate tweets (by ID, just in case)
#Expand nested object series (place)
#Expand place bounding box
#Extract user id and user locations
#Drop duplicate users
#Concat extended_tweet and tweet info for full text column
#Drop tweets mentioning CVS health - lots of hiring tweets!

atdf = atdf.rename(columns={'coordinates':'precise_coor'})
atdf['created_at'] = pd.to_datetime(atdf.created_at)
atdf['source'] = atdf['source'].str.extract('>(.+?)<', expand=False).str.strip() 
atdf = atdf[atdf['place'].notna()]
atdf = atdf.drop_duplicates(subset='id_str', keep="first")
atdf = pd.concat([atdf.drop(['place'], axis=1), atdf['place'] \
                  .apply(pd.Series)], axis=1)
atdf = pd.concat([atdf.drop(['bounding_box'], axis=1), atdf['bounding_box'] \
                  .apply(pd.Series)], axis=1)
atdf = pd.concat([atdf, atdf['user'].apply(pd.Series)\
      .rename(columns={'id_str': 'userid', 'location': 'userlocation'})\
      [['userid','userlocation']]], axis=1)
atdf = atdf.drop_duplicates(subset='userid', keep="first")
atdf = pd.concat([atdf, atdf['extended_tweet'].apply(pd.Series)\
      ['full_text']], axis=1)
atdf['full_text'] = atdf['full_text'].fillna(atdf['text'])
atdf = atdf[~atdf.full_text.str.contains("CVS Health")]
print(atdf.info())
  
####Map sentiment to dataframe
#Initiate Naive Bayes Analyzer Blobber
tb = Blobber(analyzer=NaiveBayesAnalyzer())

#Function to return sentiments

def sentiments(x):
    analysis = tb(x)
    analysiscls = analysis.sentiment.classification
    analysispos = analysis.sentiment.p_pos
    analysisneg = analysis.sentiment.p_neg
    tbpolarity = TextBlob(x).sentiment.polarity
    return analysiscls, analysispos, analysisneg, tbpolarity

#Apply function to dataframe

atdf['sentclass'], atdf['sentpos'], atdf['sentneg'], atdf['tbpolarity'] = \
    zip(*atdf['full_text'].map(sentiments))

####Map centroid to dataframe (calculated from place bounding box)
#Define point class

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
     
#Centroid calculating formula 
#http://www.samuelbosch.com/2014/05/working-in-lat-long-great-circle.html?m=1

def get_centroid(points):
    if len(points) <= 1:
        points = points[0]
    sum_x,sum_y,sum_z = 0,0,0
    for p in points:
        if not isinstance(p,Point):
            p = Point(p[0],p[1])
        lat = radians(p.y)
        lon = radians(p.x)
        ## convert lat lon to cartesian coordinates
        sum_x = sum_x + cos(lat) * cos(lon)
        sum_y = sum_y + cos(lat) * sin(lon)
        sum_z = sum_z + sin(lat)
    avg_x = sum_x / float(len(points))
    avg_y = sum_y / float(len(points))
    avg_z = sum_z / float(len(points))
    center_lon = atan2(avg_y,avg_x)
    hyp = sqrt(avg_x*avg_x + avg_y*avg_y) 
    center_lat = atan2(avg_z, hyp)
    return Point(degrees(center_lon), degrees(center_lat))

#Apply centroid function to dataframe

def returnlat(point):
    return point.y

def returnlon(point):
    return point.x

atdf['bbcentroid'] = atdf['coordinates'].map(get_centroid)
atdf['bbcenlat'] = atdf['bbcentroid'].map(returnlat)
atdf['bbcenlon'] = atdf['bbcentroid'].map(returnlon)

####Create seperate US dataframes
usdf = atdf.query('country_code == "US"')
print(usdf.info())

#Extract state name from usdf - Map State name to frame
stabv = {"ALABAMA":"AL","ALASKA":"AK","AMERICAN SAMOA":"AS","ARIZONA":"AZ",
         "ARKANSAS":"AR","CALIFORNIA":"CA","COLORADO":"CO","CONNECTICUT":"CT",
         "DELAWARE":"DE","DISTRICT OF COLUMBIA":"DC","FLORIDA":"FL",
         "GEORGIA":"GA","GUAM":"GU","HAWAII":"HI","IDAHO":"ID","ILLINOIS":"IL",
         "INDIANA":"IN","IOWA":"IA","KANSAS":"KS","KENTUCKY":"KY",
         "LOUISIANA":"LA","MAINE":"ME","MARYLAND":"MD","MASSACHUSETTS":"MA",
         "MICHIGAN":"MI","MINNESOTA":"MN","MISSISSIPPI":"MS","MISSOURI":"MO",
         "MONTANA":"MT","NEBRASKA":"NE","NEVADA":"NV","NEW HAMPSHIRE":"NH",
         "NEW JERSEY":"NJ","NEW MEXICO":"NM","NEW YORK":"NY",
         "NORTH CAROLINA":"NC","NORTH DAKOTA":"ND","NORTHERN MARIANA IS":"MP",
         "OHIO":"OH","OKLAHOMA":"OK","OREGON":"OR","PENNSYLVANIA":"PA",
         "PUERTO RICO":"PR","RHODE ISLAND":"RI","SOUTH CAROLINA":"SC",
         "SOUTH DAKOTA":"SD","TENNESSEE":"TN","TEXAS":"TX","UTAH":"UT",
         "VERMONT":"VT","VIRGINIA":"VA","VIRGIN ISLANDS":"VI",
         "WASHINGTON":"WA","WEST VIRGINIA":"WV","WISCONSIN":"WI","WYOMING":"WY"}

#Function to cleanup Twitter's nonsense
def exstate(text):
    textlist = text.split(",")
    textlist = [x.strip(' ') for x in textlist]
    if (textlist[-1] == "USA"):
        state = textlist[0].upper()
        if state in stabv: 
            state = stabv[state]
        else:
            state = "ZZ"
    else:
        state = textlist[-1]
    if len(state) > 2:
        state = "ZZ"
    return state
    
usdf['statename'] = usdf['full_name'].map(exstate)

print(usdf['statename'])

#Create second US dataframe without admin coordinates (NO STATE CENTROIDS)
#This is primarily for the HDBSCAN analysis

usdfp = usdf.query('place_type == "city"')

#Select non-null state names (territories and points of interest dropped)
#Only applies to the original US dataframe
usdf = usdf.query('statename != "ZZ"')
print(usdfp.info())

#################################ANALYSIS#####################################
####HDBSCAN Setup
#Best used with usdfp dataframe as it drops non-city level coordinates
#Nearest Neighbors Plot

setupngbrs = NearestNeighbors(n_neighbors=2)
nearstnbrs = setupngbrs.fit(usdfp[['bbcenlat','bbcenlon']])

distances, indices = nearstnbrs.kneighbors(usdfp[['bbcenlat','bbcenlon']])

distances = np.sort(distances, axis=0)
distances = distances[:,1]

fig1 = plt.figure(figsize=(10,10))
plt.plot(distances)
fig1.suptitle('Nearest Neighbors of US Vaccine Tweets', fontsize=20)
plt.xlabel('Index', fontsize=16)
plt.ylabel('Dec. Deg Distance', fontsize=16)
#fig1.savefig('Nearest Neighbors.jpg')

#Conduct DBSCAN and pull clusters (labels)
#dbsc = DBSCAN(eps = 0.5, min_samples = 30).fit(usdfp[['bbcenlat','bbcenlon']])
hdbsc = hdbscan.HDBSCAN(min_cluster_size = 30).fit(usdfp[['bbcenlat','bbcenlon']])
#dbsc = HDBSCAN(eps = 0.5, min_samples = 30).fit(usdfp[['bbcenlat','bbcenlon']])
clusters = hdbsc.labels_
usdfp['cluster'] = hdbsc.labels_

#Cluster counts as list
clcounts = np.bincount(clusters[clusters>=0])
print(clcounts)

#Setup random color generator (noise = gray, always last) and vectorizer

colorlist = []
countcolors = {}
for i in range(len(clcounts)-1):
    color = "#%06x" % random.randint(0, 0xFFFFFF)
    colorlist.append(color)
    countcolors[clcounts[i]] = color
colorlist.append("#5d6e70")

vectorizer = np.vectorize(lambda x: colorlist[x % len(colorlist)])

# Number of clusters in labels, ignoring noise, and array of cluster counts
nclusters = len(set(clusters)) - (1 if -1 in clusters else 0)
nnoise = list(clusters).count(-1)

print('Cluster counts: ', countcolors)
print('Estimated number of clusters: %d' % nclusters)
print('Estimated number of noise points: %d' % nnoise)

print(usdfp['cluster'])

#Plot our results

fig2 = plt.figure(figsize=(10,10))
plt.scatter(usdfp['bbcenlon'],
            usdfp['bbcenlat'],
            c=vectorizer(clusters))
fig2.suptitle("HDBSCAN US Vaccine Tweets", fontsize=20)
plt.xlabel("Longitude", fontsize=16)
plt.ylabel("Latitude", fontsize=16)
fig2.savefig('HDBSCAN of US Vaccine Tweets Clusters.jpg')


####Map state-level vaccine sentiment
#Create state sentiment means df
#Merges with state pop, covid, vaccine data for later use

statedf = usdf.groupby('statename') \
       .agg(count=('full_text', 'size'), sentpos=('sentpos', 'mean')) \
       .reset_index().rename(columns = {'statename':'Shorthand'})
supldf = pd.read_csv("covid_statedata.csv")
statedf = pd.merge(statedf, supldf, on="Shorthand")
print(statedf)

countdf = usdf["statename"].value_counts()
print(usdf["statename"].value_counts())
print(statedf.info())


#Function to create AK and HI maps
def add_insetmap(axes_extent, map_extent, state_name, facecolor, edgecolor, geometry):
    # create new axes, set its projection
    use_projection = ccrs.Mercator()     # preserve shape well
    #use_projection = ccrs.PlateCarree()   # large distortion in E-W for Alaska
    geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum='WGS84'))
    sub_ax = plt.axes(axes_extent, projection=use_projection)  # normal units
    sub_ax.set_extent(map_extent, geodetic)  # map extents

    # add basic land, coastlines of the map
    sub_ax.add_feature(cartopy.feature.LAND)
    sub_ax.coastlines()

    sub_ax.set_title(state_name)

    # add map `geometry` here
    sub_ax.add_geometries([geometry], ccrs.PlateCarree(), \
                          facecolor=facecolor, edgecolor=edgecolor)
    # +++ more features can be added here +++

    # plot box around the map
    extent_box = sgeom.box(map_extent[0], map_extent[2], map_extent[1], map_extent[3])
    sub_ax.add_geometries([extent_box], ccrs.PlateCarree(), color='none', linewidth=0.05)

#Start creating out figure
fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal())
ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())
shapename = 'admin_1_states_provinces_lakes_shp'
states_shp = shpreader.natural_earth(resolution='110m',
                                     category='cultural', name=shapename)
ax.background_patch.set_visible(False)
ax.outline_patch.set_visible(False)
mcolors = ["#FF6962","#ffcfcf","#F3E7CE",
               "#c2edd1","#5FA777"]
pt1 = mpatches.Patch(color=mcolors[0], label='Negative')
pt2 = mpatches.Patch(color=mcolors[1], label='Slightly Negative')
pt3 = mpatches.Patch(color=mcolors[2], label='Neutral')
pt4 = mpatches.Patch(color=mcolors[3], label='Slightly Positive')
pt5 = mpatches.Patch(color=mcolors[4], label='Positive')
plt.legend(handles=[pt1,pt2,pt3,pt4,pt5],
           loc='lower right',fontsize=14,bbox_to_anchor=(0.9,-.3))
ax.set_title('State vaccine sentiment', fontsize=20)


for state in shpreader.Reader(states_shp).records():
    edgecolor = 'black'
    stateabv = state.attributes['postal']
    statesent = statedf.loc[statedf["Shorthand"] == stateabv]['sentpos'].values[0]
    # simple scheme to assign color to each state
    if statesent < 0.4:
        facecolor = mcolors[0]
    elif statesent >= 0.4 and statesent < 0.475:
        facecolor = mcolors[1];
    elif statesent >= 0.475 and statesent < 0.525:
        facecolor = mcolors[2];
    elif statesent >= 0.525 and statesent < 0.60: 
        facecolor = mcolors[3];
    elif statesent >= 0.60:
        facecolor = mcolors[4];
    else:
        facecolor = "gray"

    # special handling for AK and HI
    if state.attributes['name'] in ("Alaska", "Hawaii"):
        # print("state.attributes['name']:", state.attributes['name'])

        state_name = state.attributes['name']

        # prep map settings
        # experiment with the numbers in both `_extents` for your best results
        if state_name == "Alaska":
            # (1) Alaska
            map_extent = (-178, -135, 46, 73)    # degrees: (lonmin,lonmax,latmin,latmax)
            axes_extent = (0.04, 0.06, 0.29, 0.275) # axes units: 0 to 1, (LLx,LLy,width,height)

        if state_name == "Hawaii":
            # (2) Hawii
            map_extent = (-162, -152, 15, 25)
            axes_extent = (0.27, 0.06, 0.15, 0.15)

        # add inset maps
        add_insetmap(axes_extent, map_extent, state_name, \
                     facecolor, \
                     edgecolor, \
                     state.geometry)

    # the other (conterminous) states go here
    else:
        # `state.geometry` is the polygon to plot
        ax.add_geometries([state.geometry], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=edgecolor)

plt.show()

####Regression of state-level vaccine sentiment
#Use statedf 

####Make Histogram
plt.rcParams["figure.figsize"] = [10,6]
plt.rcParams.update({'font.size': 16})

plt.hist(statedf['sentpos'])
plt.title('Histogram of State Positive Sentiment Values')
plt.xlabel("Tweet Positive Sentiment Score")
plt.ylabel("Frequency")
plt.grid()
#plt.savefig("fig2_histogram_v1.png")

#### Run Linear Regression

#Function to send regression table results to Pandas DF

def ols2pandas(mydf,results):
    for mod in results.keys():
        for col in results[mod].tables[0].columns:
            if col % 2 == 0: 
                mydf = mydf.append(pd.DataFrame({'Model': [mod]*results[mod].tables[0][col].size,
                                             'Param':results[mod].tables[0][col].values, 
                                             'Value':results[mod].tables[0][col+1].values}))
    return mydf;

#Function to run regression on full dataset or by column var
#Results get added to a dictionary to later be processed
    
def sepregress(dataframe,sepvar,dmatquer,allresults):
    if sepvar == None:
        y, X = dmatrices(dmatquer,data=dataframe, return_type='dataframe')
        mod = sm.OLS(y, X)
        res = mod.fit()
        resummary = res.summary2()
        allresults["Full DF"] = resummary
    elif sepvar is not None:
        sepvalues = dataframe[sepvar].unique()
        cquery = 'intvar == "intval"'.replace('intvar',sepvar)
        for i in sepvalues:
            zquery = cquery.replace('intval',i)
            cdf = dataframe.query(zquery)
            y, X = dmatrices(dmatquer,data=cdf, return_type='dataframe')
            mod = sm.OLS(y, X)
            res = mod.fit()
            resummary = res.summary2()
            allresults[i] = resummary
  
#Run our regression  
print(statedf.info())
modresults = {}
sepregress(statedf.query('count > 20'),
           None,
           'doses18yo100kmay3 ~ urbanpop',
           modresults)
print(modresults)


####Return to our clustering...

usdfp['clustersent'] = usdfp.groupby('cluster')['sentpos'].transform('mean')
print(usdfp.groupby(['cluster', 'clustersent']).size().reset_index(name='Freq'))

def assigncolor(value):
    if value < 0.4:
        facecolor = mcolors[0]
    elif value >= 0.4 and value < 0.475:
        facecolor = mcolors[1];
    elif value >= 0.475 and value < 0.525:
        facecolor = mcolors[2];
    elif value >= 0.525 and value < 0.60: 
        facecolor = mcolors[3];
    elif value >= 0.60:
        facecolor = mcolors[4];
    else:
        facecolor = "gray"
    return facecolor

usdfp['clcolor'] = usdfp['clustersent'].map(assigncolor)

print(usdfp.info())

fig5 = plt.figure(figsize=(10,10))
plt.scatter(usdfp['bbcenlon'],
            usdfp['bbcenlat'],
            c=usdfp['clcolor'],
            edgecolors='lightgray')
fig5.suptitle("HDBSCAN US Vaccine Tweets - Colored by Sentiment", fontsize=20)
plt.xlabel("Longitude", fontsize=16)
plt.ylabel("Latitude", fontsize=16)
fig5.savefig('HDBSCAN of US Vaccine Tweets Sentiment Clusters.jpg')


################################MISC. TASKS####################################
####Dataframe info, exports, boring stoff
#Show freqs of variables of interest
    
def freqdf(df,freqvar):
    itemcounts = df[freqvar].value_counts(normalize=True)
    print(itemcounts)

#freqdf(atdf.query('country_code =="US"'),'sentclass')
print(usdf["statename"].value_counts())
print(usdf[["statename","sentpos"]].groupby("statename").mean().quantile([0.25,0.5,0.75]))

#Print info about dataframe
print(usdf.info())

#Export to CSV
atdf.to_csv('csvtest.csv')
usdf.to_csv('uscsvtest.csv')
usdfp.to_csv('uscsvtestp.csv')

