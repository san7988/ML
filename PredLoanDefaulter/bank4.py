import pandas as  pd
import numpy as np
import xgboost as xg
from sklearn.preprocessing import LabelEncoder as le
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

print "Reading Data..."
X = pd.read_csv("train_indessa.csv")
X_validate = pd.read_csv("test_indessa.csv")

members_train = X["member_id"]
members_validate = X_validate["member_id"]


#X.fillna("00")
#X_validate.fillna("00")
col_to_drop = ["funded_amnt","funded_amnt_inv","collection_recovery_fee",\
				"desc","pymnt_plan","verification_status_joint",\
				"title","emp_title","member_id"]

col_to_encode = ["batch_enrolled","addr_state","zip_code","initial_list_status","last_week_pay","grade","sub_grade","purpose",\
				"verification_status","home_ownership","application_type","acc_now_delinq",\
				"tot_coll_amt","collections_12_mths_ex_med","delinq_2yrs", "inq_last_6mths",\
				 "pub_rec", "total_rec_late_fee","recoveries","term"]
X.drop(col_to_drop,axis=1,inplace=True)
X_validate.drop(col_to_drop,axis=1,inplace=True)

#print X["term"].unique()
X["term"] = X["term"].str[:2].astype(int)
X_validate["term"] = X_validate["term"].str[:2].astype(int)


#X[X["emp_length"]=="n/a","emp_length"] = "-1"
X.ix[X.emp_length == "n/a","emp_length"] = "-1"
X.ix[X.emp_length == "< 1 year","emp_length"] = "00"
X.ix[X.emp_length == "10+ years","emp_length"] = "10"
X["emp_length"] = X["emp_length"].str[:2].astype(int)
#print X["emp_length"][0:10]

X_validate.ix[X_validate.emp_length == "n/a","emp_length"] = "-1"
X_validate.ix[X_validate.emp_length == "< 1 year","emp_length"] = "00"
X_validate.ix[X_validate.emp_length == "10+ years","emp_length"] = "10"
X_validate["emp_length"] = X_validate["emp_length"].str[:2].astype(int)


#print X["annual_inc"].unique()
#X[X["annual_inc"]=="00"] = 0
X["annual_inc"] = X["annual_inc"].astype(float) + 10
X["annual_inc"] = X["annual_inc"].astype(float).apply(np.log) + 10

#X_validate[X_validate["annual_inc"]=="00"] = 0
X_validate["annual_inc"] = X_validate["annual_inc"].astype(float) + 10
X_validate["annual_inc"] = X_validate["annual_inc"].astype(float).apply(np.log) + 10

# print X["annual_inc"].unique()
# print X["annual_inc"].isnull().any().any()

col_to_check_skew = ["int_rate","dti","mths_since_last_delinq",\
				"mths_since_last_record","open_acc","revol_bal",\
				"revol_util","total_acc","total_rec_int","mths_since_last_major_derog","tot_cur_bal",\
				"total_rev_hi_lim"]
print "In train data..."
for col in col_to_check_skew:
	X[col].fillna(0,inplace=True)
	X[col] = X[col].astype(float)
	skew = X[col].skew()
	print "for ",col," skew is :",skew
	if skew >2 :
		X[col] = X[col] + 10
		X[col] = X[col].apply(np.log) + 10

print "In validate data..."
for col in col_to_check_skew:
	X_validate[col].fillna(0,inplace=True)
	X_validate[col] = X_validate[col].astype(float)
	skew = X_validate[col].skew()
	print "for ",col," skew is :",skew
	if skew >2 :
		X_validate[col] = X_validate[col] + 10
		X_validate[col] = X_validate[col].apply(np.log) + 10

X[:100].to_csv("X.csv",index=False)
# for col in X.columns:
# 	print col
# 	print X[col].dtype
# 	print X[col].isnull().any().any()
# 	print X[col].unique()
# sys.exit(1)

l = le()
for col in col_to_encode:
	X[col] = l.fit_transform(X[col])
	X_validate[col] = l.fit_transform(X_validate[col])

X["inc_by_amnt"] = (X["annual_inc"].astype(float)/X["loan_amnt"].astype(float)).apply(np.log)
X["int_by_late_fee"] = X["total_rec_int"].astype(float) + X["total_rec_late_fee"].astype(float)
X["amnt_*_int"] = (X["loan_amnt"].astype(float)*X["int_rate"].astype(float)).apply(np.sqrt)
X.insert(0, 'payment_completion', (X['last_week_pay']/(X['term']/12*52+1))*100)
X['payment_completion'] = X['payment_completion'].astype(int)
X[:100].to_csv("X.csv",index=False)
#sys.exit(1)
X_validate["inc_by_amnt"] = (X_validate["annual_inc"].astype(float) / \
					X_validate["loan_amnt"].astype(float)).apply(np.log)
X_validate["int_by_late_fee"] = X_validate["total_rec_int"].astype(float) +\
					 X_validate["total_rec_late_fee"].astype(float)
X_validate["amnt_*_int"] = (X_validate["loan_amnt"].astype(float) * \
					X_validate["int_rate"].astype(float)).apply(np.sqrt)
X_validate.insert(0, 'payment_completion', (X_validate['last_week_pay']/(X_validate['term']/12*52+1))*100)
X_validate['payment_completion'] = X_validate['payment_completion'].astype(int)

Y = X["loan_status"].astype(int)
del X["loan_status"]
Y = pd.DataFrame(Y)
X["loan_amnt"] = X["loan_amnt"].astype(float) 
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,\
									 test_size=test_size, random_state=seed)
# for col in X.columns:
# 	print col
# 	print X[col].dtype
# 	print X[col].isnull().any().any()
# 	print X[col].unique()
#del X["loan_status"]
print "Finished Reading Data...Now creating train test data..."
print "Starting Prediction..."

model = xg.XGBClassifier(base_score=0.5, colsample_bytree=0.9, gamma=0,
	       learning_rate=0.3, max_delta_step=0, max_depth=6,
	       min_child_weight=1, missing=None, n_estimators=150, nthread=4,
	       objective='binary:logistic', seed=29, silent=False, subsample=0.9)

model.fit(X_train, np.array(Y_train).ravel(),eval_metric='auc')
print model

Y_pred = model.predict(X_test)
print Y_pred,type(Y_pred)
Y_predprob = model.predict_proba(X_test)[:,1]
print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape
print Y_pred.shape
print Y_predprob.shape

predictions = [round(float(value)) for value in Y_pred]
accuracy = accuracy_score(np.array(Y_test).ravel(), predictions)
print "Accuracy: %.2f%%" % (accuracy * 100.0)
print "AUC Score (Test): %f" % metrics.roc_auc_score(Y_test, Y_predprob)
print "*************************************"
Y_validate = model.predict_proba(X_validate)[:,1]
submission = pd.concat([pd.DataFrame(members_validate),pd.DataFrame(Y_validate)],axis=1)
print submission[0:10]
submission.columns = ["member_id","loan_status"]
submission.to_csv("submission.csv",index=False)
