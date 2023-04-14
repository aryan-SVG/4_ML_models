from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
#Decision Trees
from sklearn.tree import DecisionTreeRegressor
#K-nearest neighbour
from sklearn.neighbors import KNeighborsRegressor
#support-vector
from sklearn.svm import SVR
import pandas as pd
#Bagged regressor
from sklearn.ensemble import BaggingRegressor
#Boosted regressr
from sklearn.ensemble import AdaBoostRegressor
#voting Regressor
from sklearn.ensemble import VotingRegressor
 
lis = datasets.fetch_openml(data_id=688)
#lis = datasets.fetch_openml(data_id=34615)

lis.data.info()

lis.data["isns"].unique()

ct = ColumnTransformer([("encoder",
OneHotEncoder(sparse=False), [3])],
remainder="passthrough")

new_data = ct.fit_transform(lis.data)
ct.get_feature_names_out()    


lis_new_data = pd.DataFrame(new_data, columns =
ct.get_feature_names_out(), index = lis.data.index)
lis_new_data.info()

lis_new_data

#liner Regression-normal 
lr = LinearRegression()
lr_scores = cross_validate(lr, lis_new_data, lis.target,
cv=10, scoring="neg_root_mean_squared_error")

lr_scores["test_score"]

lr_rmse = 0-lr_scores["test_score"]
lr_rmse.mean()
    

# bagged liner Regression 
bagged_lr = BaggingRegressor(LinearRegression())
bagged_lr_scores = cross_validate(bagged_lr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

bagged_lr_scores["test_score"]

bagged_lr_rmse = 0-bagged_lr_scores["test_score"]
bagged_lr_rmse.mean()
    

#boosted liner regressor 
#from sklearn.ensemble import AdaBoostRegressor
boosted_lr = AdaBoostRegressor(LinearRegression())
boosted_lr_scores = cross_validate(boosted_lr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error") 


boosted_lr_scores["test_score"]

boosted_lr_rmse = 0-bagged_lr_scores["test_score"]
boosted_lr_rmse.mean()
    

# K-nearest-normal

parameters = [{"n_neighbors":[3,5,7,9,11,13,15]}]
tuned_knn = GridSearchCV(KNeighborsRegressor(), parameters, scoring="neg_root_mean_squared_error", cv=10)

kn_scores = cross_validate(tuned_knn, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

kn_scores["test_score"]

kn_rmse = 0-kn_scores["test_score"]
kn_rmse.mean()

#bagged K-nearest 
bagged_kn = BaggingRegressor(tuned_knn)
bagged_kn_scores = cross_validate(bagged_kn, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

bagged_kn_scores["test_score"]

bagged_kn_rmse = 0-bagged_kn_scores["test_score"]
bagged_kn_rmse.mean()

#boosted K-nearest 
boosted_kn = AdaBoostRegressor(tuned_knn)
boosted_kn_scores = cross_validate(boosted_kn, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

boosted_kn_scores["test_score"]


boosted_kn_rmse = 0-boosted_kn_scores["test_score"]
boosted_kn_rmse.mean()

#Decision Trees
dtc = DecisionTreeRegressor()
tuned_dtc =GridSearchCV(dtc, [{'min_samples_leaf':[1,5,10,15]}], scoring="neg_root_mean_squared_error", cv=10)
dtc_score =cross_validate(tuned_dtc,
lis.data, lis.target, scoring="neg_root_mean_squared_error", cv=10)

dtc_score["test_score"]

dtc_rmse = 0-dtc_score["test_score"]
dtc_rmse.mean()

#Decision Tree-Bagged
bagged_dtc = BaggingRegressor(tuned_dtc)
bagged_dtc_scores = cross_validate(bagged_dtc, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")


bagged_dtc_scores["test_score"]

bagged_dtc_rmse = 0-bagged_dtc_scores["test_score"]
bagged_dtc_rmse.mean()

#Decision Tree-Boosted
boosted_dtc = AdaBoostRegressor(tuned_dtc)
boosted_dtc_scores = cross_validate(boosted_dtc, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

boosted_dtc_scores["test_score"]

boosted_dtc_rmse = 0-boosted_dtc_scores["test_score"]
boosted_dtc_rmse.mean()

#SVM- normal 
svm = SVR()
svm_scores = cross_validate(svm, lis_new_data, lis.target,
cv=10, scoring="neg_root_mean_squared_error")

svm_scores["test_score"]

svm_rmse = 0-svm_scores["test_score"]
svm_rmse.mean()
    

#svm- bagged
bagged_svm = BaggingRegressor(SVR())
bagged_svm_scores = cross_validate(bagged_svm, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

bagged_svm_scores["test_score"]

bagged_svm_rmse = 0-bagged_svm_scores["test_score"]
bagged_svm_rmse.mean()

#svm- boosted
boosted_svm = AdaBoostRegressor(SVR())
boosted_svm_scores = cross_validate(boosted_svm, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

boosted_svm_scores["test_score"]

boosted_svm_rmse = 0-boosted_svm_scores["test_score"]
boosted_svm_rmse.mean()

from sklearn.ensemble import VotingRegressor
vr = VotingRegressor([("lr", LinearRegression()), ("svm", SVR()),("dtc", tuned_dtc),("knn", tuned_knn)])
voting_score=cross_validate(vr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")

voting_score["test_score"]

voting_rmse = 0-voting_score["test_score"]
voting_rmse.mean()

from scipy.stats import ttest_rel
#p-value base & bagged
#Linear regression
ttest_rel(lr_rmse, bagged_lr_rmse)

#p-value base & bagged
#Decision trees 
ttest_rel(dtc_rmse, bagged_dtc_rmse)

#p-value base & bagged
#K-nearest 
ttest_rel(kn_rmse, bagged_kn_rmse)

#p-value base & bagged
#SVM
ttest_rel(svm_rmse, bagged_svm_rmse)

#p-vale base & boosted
#liner regression
ttest_rel(lr_rmse, boosted_lr_rmse)

#p-value base & boosted
#Decision trees 
ttest_rel(dtc_rmse, boosted_dtc_rmse)

#p-value base & boosted
#K-nearest 
ttest_rel(kn_rmse, boosted_kn_rmse)

#p-value base & boosted
#SVM
ttest_rel(svm_rmse, boosted_svm_rmse)


#####################################
##################################
#############################




#p value base & decision
#Linear-regression
ttest_rel(lr_rmse,dtc_rmse)

#p value base & decision
#voting 
ttest_rel(voting_rmse, dtc_rmse)

#p value base & decision
#K-nearest 
ttest_rel(kn_rmse, dtc_rmse)

#p value base & decision
#SVM
ttest_rel(svm_rmse,dtc_rmse )

