def modelOptimization(self):	
		# param_test1 = {
		# 	'max_depth':range(3,10,2),
		# 	'min_child_weight':range(1,6,2)
		# }
		# gsearch1 = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
		# 	min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
		# 	objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
		# 	param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
		# gsearch1.fit(X_train,y_train)
		# print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_	
		# param_test2 = {
		#  'max_depth':[6,7,8],
		#  'min_child_weight':[2,3,4]
		# }
		# gsearch2 = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
		#  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
		#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
		#  param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
		# gsearch2.fit(X_train,y_train)
		# print gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

		# param_test2b = {
		#  'min_child_weight':[2,6,8,10,12]
		# }
		# gsearch2b = GridSearchCV(estimator = xgboost.XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
		#  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
		#  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
		#  param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
		# gsearch2b.fit(X_train,y_train)	
		# print gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_
		pass	