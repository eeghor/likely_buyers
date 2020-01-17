from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, Normalizer
import arrow
import numpy as np
import json
import time

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer


outbound_trips = json.load(open('data/outbound_trips.json'))
cnt_lefthand = pd.read_csv('data/countries_lefthand.csv')

def competitors_price(company, country, days, cheap=True):

	"""
	returns a quote for extra insurance from selected rental companies; the amount is in USD
	"""
	
	quote = {'au': 

				{'thrifty': lambda days, cheap: (days <= 10)*(31.35*cheap + 39.60*(1.0 - cheap)) + \
												(days > 10)*(31.35*cheap + round(396.0/days,2)*(1.0 - cheap)),

				 # this is Excess Reduction, last checked 16/01/2020
				 'avis': lambda days, cheap: (days <= 10)*27.0 + \
				 							 (11 <= days <= 27)*round(270.0/days,2) + \
				 							 (days > 27)*6.0,

				 'herz': lambda days, cheap: 26.99*cheap + 40.0*(1.0 - cheap),

				 'europcar': lambda days, cheap: (days == 1)*(49.49*cheap + 59.14*(1.0 - cheap)) + \
												 (2 <= days <= 3)*(37.41*cheap + 47.07*(1.0 - cheap)) + \
												 (4 <= days <= 6)*(34.40*cheap + 43.44*(1.0 - cheap)) + \
												 (days > 6)*(26.55*cheap + 36.20*(1.0 - cheap)),
												 
				 'budget': lambda days, cheap: (days < 10)*27.0 + (10 <= days <=27)*round(270.0/days,2) + \
											   (days >= 28)*6.0},

			'gb': 

				{'thrifty': lambda days, cheap: (days < 4)*(8.81*cheap + 15.0*(1.0 - cheap)) + \
												(4 <= days < 14)*(7.15*cheap + 11.5*(1.0 - cheap)) + \
												(days >= 14)*(6.05*cheap + 9.05*(1.0 - cheap)),
				 'europcar': lambda days, cheap: (days < 4)*(8.81*cheap + 15.0*(1.0 - cheap)) + \
												 (4 <= days < 14)*(7.15*cheap + 11.5*(1.0 - cheap)) + \
												 (days >= 14)*(6.05*cheap + 9.05*(1.0 - cheap)),
				 'herz': lambda days, cheap: 26.40,
				 'sixt': lambda days, cheap: (days < 6)*(13.0*cheap + 28.5*(1.0 - cheap)) + \
											 (days == 6)*(10.99*cheap + 24.5*(1.0 - cheap)) + \
											 (days == 7)*(10.49*cheap + 23.5*(1.0 - cheap)) + \
											 (8 <= days <= 14)*(10.0*cheap + 21.40*(1.0 - cheap)) + \
											 (days > 14)*(10.0*cheap + 20.30*(1.0 - cheap))
				},

			'es':

				{'europcar': lambda days, cheap: (days < 5)*(26.21*cheap + 37.21*(1.0 - cheap)) + \
												 (5 <= days <= 7)*(22.80*cheap + 31.21*(1.0 - cheap)) + \
												 (8 <= days <= 15)*(20.20*cheap + 24.32*(1.0 - cheap)) + \
												 (days > 15)*(17.18*cheap + 17.36*(1.0 - cheap))}
			}

	xch_to_usd = {'au': 0.69,
				  'gb': 1.30,
				  'es': 1.11}
	
	if not isinstance(country, str):
		return None
	
	_c = country.lower().strip()
	
	if not company:
		try:
			perday = min({quote[_c][comp](days=days, cheap=True) for comp in quote[_c]})
		except:
			return None
	else:
		try:
			perday = quote[_c][company.lower().strip()](days=days, cheap=True)
		except:
			return None
	
	return round(xch_to_usd[_c]*perday*days,2)


class ModelTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, model):
		self.model = model

	def fit(self, *args, **kwargs):
		self.model.fit(*args, **kwargs)
		return self

	def transform(self, X, **kwargs):
		return self.model.predict(X)

class ColumnAsIs(BaseEstimator, TransformerMixin):

	"""
	does nothing, simply returns a column
	"""

	def transform(self, X):
		return X

	def fit(self, X, y=None):
		return self

class CountryIsLeftHand(BaseEstimator, TransformerMixin):

	"""
	returns 1 if a country is left-hand driving and 0 otherwise
	"""

	def transform(self, X):

		countries_lefthand = ['AG', 'AU', 'BB', 'BD', 'BN', 'BS', 'BT', 'BW', 'CY', 'DM',
							  'FJ', 'GB', 'GD', 'GY', 'ID', 'IE', 'IN', 'JM', 'JP', 'KE',  
							  'KI', 'KN', 'LC', 'LK', 'LS', 'MT', 'MU', 'MV', 'MW', 'MY',  
							  'MZ', 'NA', 'NP',  
							  'NR', 'NZ', 'PG', 'PK', 'SB', 'SC', 'SG', 'SR', 
							  'SZ', 'TH', 'TO', 'TP', 'TT',  
							  'TV', 'TZ', 'UG', 'VC', 'WS', 'ZA', 'ZM', 'ZW']

		return X.isin(countries_lefthand).astype(int)

	def fit(self, X, y=None):
		return self

class CountryFrequentlyBooked(BaseEstimator, TransformerMixin):

	"""
	returns a score describing how frequently this country is booked
	"""

	def __init__(self, score_dict):
		self.score_dict = score_dict

	def transform(self, X):

		return X.iloc[:,0].apply(lambda _: self.score_dict.get(str(_), 0)).values.reshape(len(X),1)

	def fit(self, X, y=None):
		return self

class DataLoader:

	def __init__(self):

		"""
		"""
		
		self.cols_to_parse_date = []
		self.cols_to_drop = 'CustomerId BookingId Reference'.split()

	def load(self, file='rc_features_17JAN2020.csv', countries=None):

		self.data = pd.read_csv('data/' + file, 
								parse_dates=self.cols_to_parse_date, 
								dtype={'TotalUSD': float, 
										'Total': float},
								keep_default_na=False, na_values='') 

		# self.test = pd.read_csv('data/test_unfiltered.csv', parse_dates=self.cols_to_parse_date, 
		# 						dtype={'TotalUSD': float},
		# 						keep_default_na=False)
		# \
		# 						.fillna({'PrevActBooking': 2, # basically n/a
		# 								 'FirstActBooking': 2,
		# 								 'PrevActThisCnt': 2
		# 								})
		# self.data = self.data[self.data['UpfrontDays'] >= 0]

		bkcount_to = Counter(self.data[self.data['isBooking']==1]['ToCountry'])
		tot_bookings = sum(bkcount_to.values())
		self.bkscore_to = {c: round(bkcount_to[c]/tot_bookings,3) for c in bkcount_to}

		if countries:

			print(f'--- filter: only customers from {", ".join(countries)}')
			self.data = self.data[self.data['ResCountry'].isin(countries)]
			# self.test = self.test[self.test['ResCountry'].isin(countries)]

		self.data = self.data.drop_duplicates(['CustomerId', 'Reference'])

		self.data['dest_popul'] = self.data[['ResCountry', 'ToCountry']] \
										.apply(lambda x: outbound_trips[x[0]].get(x[1], 0) if x[0] in outbound_trips else 0, axis=1)
		# self.test['dest_popul'] = self.test[['ResCountry', 'ToCountry']] \
		# 								.apply(lambda x: outbound_trips[x[0]].get(x[1], 0) if x[0] in outbound_trips else 0, axis=1)

		

		self.data['savings'] =  (self.data[['ToCountry', 'DurationDays']] \
								.apply(lambda _: competitors_price(company=None, country=_[0], days=_[1]), axis=1) \
								.where(lambda _: _.notnull(), self.data['TotalUSD']*1.5) -  self.data['TotalUSD']) \
								.apply(lambda x: round(max(x,0),2))

		# self.test['savings'] =  (self.test[['ToCountry', 'DurationDays']] \
		# 						.apply(lambda _: competitors_price(company=None, country=_[0], days=_[1]), axis=1) \
		# 						.where(lambda _: _.notnull(), self.test['TotalUSD']*1.5) -  self.test['TotalUSD']) \
		# 						.apply(lambda x: round(max(x,0),2))

		self.data_summary = {'rows': len(self.data), 
							 'cids': self.data['CustomerId'].nunique(),
							 'refs': self.data['Reference'].nunique(),
							 'books': self.data[self.data['isBooking'] == 1]['Reference'].nunique(),
							 'quots': self.data[self.data['isBooking'] == 0]['Reference'].nunique()}

		self.data_summary.update({'quots,%': round(100.0*self.data_summary['quots']/self.data_summary['refs'],2)})

		for _ in self.data_summary:
			print(f'{_}: {self.data_summary[_]:,}')

		self.data = self.data.drop(self.cols_to_drop, axis=1).fillna(0)
		# self.test = self.test.drop(self.cols_to_drop, axis=1).fillna(0)

		return self

if __name__ == '__main__':
	
	dl = DataLoader().load(file='rc_features_17JAN2020.csv', countries=['GB'])

	X = dl.data[[c for c in dl.data.columns if c != 'isBooking']]
	y = dl.data['isBooking'].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.26, random_state=278, stratify=y)

	# X_train = X
	# y_train = y

	# X_test = dl.test[[c for c in dl.test.columns if c != 'isBooking']]
	# y_test = dl.test['isBooking'].values

	print(f'bookings in training/test set {sum(y_train):,}/{sum(y_test):,}')
	print(f'quotes in training/test set {len(y_train) - sum(y_train):,}/{len(y_test) - sum(y_test):,}')

	features_std = FeatureUnion([
										 
								('ct', ColumnTransformer([('trip_details', 
																ColumnAsIs(), 
																['DurationDays', 'UpfrontDays', 'Cancelled']),
														  ('quote_month', 
																OneHotEncoder(handle_unknown='ignore'), 
																['QuoteMonth']),
														  ('quote_week', 
																OneHotEncoder(handle_unknown='ignore'), 
																['QuoteWeek']),
														  ('quote_day', 
																OneHotEncoder(handle_unknown='ignore'), 
																['QuoteDay']),
														  ('res_country', 
																OneHotEncoder(handle_unknown='ignore'), 
																['ResCountry']),
														  ('to_country', 
																OneHotEncoder(handle_unknown='ignore'), 
																['ToCountry']),
														  ('tocountry_lefthand',
																CountryIsLeftHand(),
																['ToCountry']),
														  ('country_freq_booked',
																CountryFrequentlyBooked(dl.bkscore_to),
																['ToCountry']),
														  ('webs_lang',
																OneHotEncoder(handle_unknown='ignore'),
																['LanguageCode']),
														  ('rescountry_lefthand',
																CountryIsLeftHand(),
																['ResCountry']),
														  ('vehicle_type', 
																ColumnAsIs(), 
																['isCar', 'is4x4', 'isCamper', 'isMinibus', 'isMotorHome']),
														  ('dest_popul', 
																ColumnAsIs(), 
																['dest_popul']),
														  ('prev_activities', 
																ColumnAsIs(), 
																['PrevBks', 'PrevQts', 'PrevCnc',
       															  'BeforeActThisCnt', 'PerdayUSD',
       															 'BeforeTotalCnt', 'BeforeTotalCurrs', 'BeforeTotalLangs']),
														  ('prev_activities_categ',
														  		OneHotEncoder(handle_unknown='ignore'),
														  		['PrevActBooking',
														  		 'FirstActBooking',
														  		 'PrevActThisCnt']),
														  ('payment_details',
														  		Normalizer(),
																['TotalUSD']),
														  ('payment_details_local',
														  		Normalizer(),
																['Total']),
														  ('savings', 
																ColumnAsIs(), 
																['savings'])
														]))
								])

	pipe = Pipeline([('features', features_std),
					 ('feat_select', SelectKBest(chi2, k=10)),
					 ('cls', GradientBoostingClassifier())])

	# t0 = time.time()
	# pipe.fit(X_train, y_train)
	# dt = time.time() - t0

	# m, s = divmod(dt, 60)

	# print(f'time to fit: {m:02.0f}:{s:02.0f}')

	# print(pipe.named_steps['randomforest'].feature_importances_)

	# y_pred = pipe.predict(X_test)

	# pipe = make_pipeline(features, StandardScaler(), RandomForestClassifier())

	pars = {'cls__n_estimators': (200, 300),
			'cls__max_depth': (2,3),
			'feat_select__k': (10,20,50)}

	# # grid_search = GridSearchCV(pp, pars, n_jobs=2, verbose=1, cv=4)

	grid_search = GridSearchCV(pipe, pars, n_jobs=2, verbose=1, cv=4)

	grid_search.fit(X_train, y_train)

	y_pred = grid_search.predict(X_test)

	# print(f'accuracy: {accuracy_score(y_test, y_h):06.4f}')

	print(classification_report(y_test, y_pred))
	print('---- confusion matrix:')
	print(confusion_matrix(y_test, y_pred))


