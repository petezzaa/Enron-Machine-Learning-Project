#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
                 'long_term_incentive', 'restricted_stock', 'director_fees', 'shared_receipt_with_poi', "fraction_from_poi",
                 "fraction_to_poi", 'fraction_exercised_stock']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#Remove Total, Eugene Lockhart, and THE TRAVEL AGENCY IN THE PARK from data_dict
data_dict.pop('TOTAL', 0)
del data_dict['LOCKHART EUGENE E']
del data_dict['THE TRAVEL AGENCY IN THE PARK']

### Task 3: Create new feature(s)
def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    if (poi_messages == "NaN" or all_messages == "NaN"):
        fraction = 0
    else:
        fraction = float(poi_messages) / float(all_messages)

    return fraction


for name in data_dict:
    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi

    exercised_stock_options = data_point['exercised_stock_options']
    total_stock_value = data_point['total_stock_value']
    fraction_exercised_stock = computeFraction(exercised_stock_options, total_stock_value)
    data_point['fraction_exercised_stock'] = fraction_exercised_stock

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

skb = SelectKBest(k = 9)
gnb = GaussianNB()
#make pipeline with selectkbest and naive bayes
pipeline = Pipeline(steps = [('skb', skb), ('gnb', gnb)])
pipeline.fit(features, labels)

#indices for features chosen by SelectKBest
skb_feat_index = skb.get_support(indices = True)
#Make a dictionary of features with their scores
my_features_score = {}
for i in skb_feat_index:
    my_features_score[features_list[i + 1]] = skb.scores_[i]
#print my_features_score

#make a list of features from the dictionary
my_features_list = my_features_score.keys()
#put 'poi' as the first element in the list
my_features_list.insert(0, 'poi')
#print my_features_list

from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import StratifiedShuffleSplit

data = featureFormat(data_dict, my_features_list)
labels, features = targetFeatureSplit(data)

#split training and testing folds
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
for train_idx, test_idx in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )

#create naive bayes classifier
gnb = GaussianNB()
gnb.fit(features_train, labels_train)

pred = gnb.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print "Naive Bayes"
print "accuracy:",accuracy
target_names = ["Non-POI", "POI"]
print "Classification report:"
print classification_report(y_true = labels_test, y_pred = pred, target_names = target_names)

from sklearn.ensemble import RandomForestClassifier

#create random forest classifier
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print "Random Forest"
print "accuracy:",accuracy
print "Classification report:"
print classification_report(y_true = labels_test, y_pred = pred, target_names = target_names)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import GridSearchCV

#create random forest classifier with GridSearchCV to tune the algorithm
rfc = RandomForestClassifier(n_estimators = 100)
parameters = {'min_samples_split':[2, 5, 10, 20],
              'max_features':('auto', 'log2'),
              'criterion':('gini', 'entropy')
             }
clf = GridSearchCV(rfc, parameters)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
print "Random Forest w/ Tuning"
print "accuracy:",accuracy
print "Classification report:"
print classification_report(y_true = labels_test, y_pred = pred, target_names = target_names)
print "Best Parameters:",clf.best_params_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(gnb, data_dict, my_features_list)