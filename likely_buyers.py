from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.decomposition import PCA
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
import arrow
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer


class VehicleType(TransformerMixin):

	"""
	what sort of a vehicle a customer was interested in
	"""

	def transform(self, X):

		veh_type_cols = 'isCar is4x4 isCamper isMinibus isMotorHome'.split()

		return X[veh_type_cols]

	def fit(self, X, y=None):
		return self

class TripDetails(TransformerMixin):

	"""
	available trip details
	"""

	def transform(self, X):


		# tr_details_cat = 'FromDayWeek ToDayWeek'.split()
		tr_details_asis = 'DurationDays UpfrontDays Cancelled'.split()

		return X[tr_details_asis]

	def fit(self, X, y=None):
		return self

class PrevActivities(TransformerMixin):

	"""
	available trip details
	"""

	def transform(self, X):

		pa_cols = 'prev_bks prev_qts prev_cnl prev_act_bk fst_act_bk last_act_same_cnt prev_act_same_cnt prev_diff_cnt'.split()

		return X[pa_cols]

	def fit(self, X, y=None):
		return self

class ToFromCountries(TransformerMixin):

	"""
	where is a customer renting a car and what's his country or residence
	"""

	def transform(self, X):

		cnt_cols = 'ToCountry ResCountry'.split()
		all_dummy_names = sorted(['_'.join([cl.lower(), c]) for c in 'C1 C2 C3 C4 XX'.split() for cl in cnt_cols])

		_X = pd.concat([pd.get_dummies(X[c], prefix=c.lower()) for c in cnt_cols], sort=False, axis=1)

		for c in all_dummy_names:
			if c not in _X.columns:
				_X[c] = 0

		return _X[all_dummy_names]


	def fit(self, X, y=None):
		return self

class CustomerDetails(TransformerMixin):

	"""
	basic customer details
	"""

	def transform(self, X):

		cus_details_cols = ['Lang']

		X['Lang'] = X['Lang'].apply(lambda x: x if x in 'en de fr es ru jp'.split() else 'xx')


		return pd.concat([pd.get_dummies(X[c], prefix=c.lower()) for c in cus_details_cols], 
			sort=False, axis=1)

	def fit(self, X, y=None):
		return self

class PaymentDetails(TransformerMixin):

	"""
	potential or actual payment related: how much in USD was the quote, was there the permanent coupon
	"""

	def transform(self, X):

		paym_cols = 'Paid Coupon'.split()

		return X[paym_cols]

	def fit(self, X, y=None):
		return self

class QuoteTiming(TransformerMixin):

	"""
	mostly time of the quote or booking
	"""

	def transform(self, X):

		qtiming_cols = 'QuoteWeek QuoteDay QuoteHour'.split()

		return pd.concat([pd.get_dummies(X[c], prefix=c.lower()) for c in qtiming_cols], 
			sort=False, axis=1)

	def fit(self, X, y=None):
		return self

class PrevQtsThisTrip(TransformerMixin):

	"""
	extract total previous bookings for each customers (for every booking or transaction)
	"""

	def transform(self, X, **kwargs):

		prevs = [] 

		for r in X.groupby('CustomerId'):

			d = r[1].sort_values('CreatedOn')

			prev_bks = []
			prev_qts = []
			prev_cnl = []
			
			for i, cr in enumerate(d.iterrows()):

				this_day = cr[1]['CreatedOnDate']

				prev_actv_this_country = d.iloc[:i]

				if not prev_actv_this_country.empty:
 
					prev_bks_this_country = max(sum([arrow.get(c).shift(days=-2) <= arrow.get(this_day) <= arrow.get(c).shift(days=+2) 
															for c in prev_actv_this_country[prev_actv_this_country['isBooking'] == 1]['FromDate']]),
												sum([arrow.get(c).shift(days=-2) <= arrow.get(this_day) <= arrow.get(c).shift(days=+2) 
															for c in prev_actv_this_country[prev_actv_this_country['isBooking'] == 1]['ToDate']]))

				else:
					prev_bks_this_country = 0

				prev_bks.append(prev_bks_this_country)

			prevs.append(pd.DataFrame({'prev_qts_same_trip': prev_bks}).set_index(d.index))

		return pd.concat(prevs)

	def fit(self, X, y=None, **kwargs):
		return self

class DataLoader:

	def __init__(self):

		"""
		'isBooking', 'FromDate',
       'ToDate', 'FromDayWeek', 'ToDayWeek', 'CreatedOn', 'CreatedOnDate',
       'Cancelled', 'DurationDays', 'UpfrontDays',  'CurrencyId', 'Paid', 'Coupon', 'isCar', 'is4x4',
       'isCamper', 'isMinibus', 'isMotorHome', 'ToCountry', 'Lang',
       'FirstName', 'LastName', 'Email', 'ResCountry', 
		"""
		
		self.cols_to_parse_date = 'FromDate ToDate CreatedOn CreatedOnDate'.split()
		self.cols_to_drop = 'CustomerId BookingId Reference'.split()

	def load(self, file='B2C_Rentalcover_04JAN2020.csv'):

		self.data = pd.read_csv('data/' + file, parse_dates=self.cols_to_parse_date)

		print(f'{len(self.data):,} rows')
		print(f'{self.data["CustomerId"].nunique():,} customer ids')
		print(f'{self.data["Reference"].nunique():,} references')
		print(f'{len(self.data[self.data["isBooking"] == 1]):,} bookings')
		print(f'{len(self.data[self.data["isBooking"] == 0]):,} quotes')

		self.data = self.data.drop(self.cols_to_drop, axis=1)

		print(list(self.data))

		cs = {'C1': ['AU', 'NZ', 'US', 'GB', 'CA', 'IE'],
				'C2': ['IT', 'ES', 'FR', 'PT', 'GR', 'CY', 'HR', 'ME', 'TR'],
				'C3': ['UA', 'RU', 'MD', 'BY', 'PL', 'CZ', 'HU'],
				'C4': ['DE', 'CH', 'NL', 'SE', 'AT', 'FI', 'NO', 'DK']}

		self.data['ToCountry'] = self.data['ToCountry'].apply(lambda x: '-'.join([c for c in cs if x in cs[c]])).apply(lambda x: x if x != '' else 'XX')
		self.data['ResCountry'] = self.data['ResCountry'].apply(lambda x: '-'.join([c for c in cs if x in cs[c]])).apply(lambda x: x if x != '' else 'XX')

		return self

if __name__ == '__main__':
	
	dl = DataLoader().load()

	X = dl.data[[c for c in dl.data.columns if c != 'isBooking']]
	y = dl.data['isBooking'].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=278, stratify=X['ToCountry'])

	print(f'bookings in training/test set {sum(y_train):,}/{sum(y_test):,}')

	features = FeatureUnion([('trip_details', TripDetails()),
							 ('to_from_countries', ToFromCountries()), 
							 ('prev_activities', PrevActivities()),
							 ('vehicle_type', VehicleType()), 
							 ('cust_details', CustomerDetails()), 
							 ('payment_details', PaymentDetails()), 
							 # ('quote_timing', QuoteTiming())
							 ])

	pipe = make_pipeline(features, StandardScaler(), RandomForestClassifier())

	pars = {'randomforestclassifier__n_estimators': (50, 100, 150, 200)}

	grid_search = GridSearchCV(pipe, pars, n_jobs=2, verbose=1, cv=4)

	grid_search.fit(X_train, y_train)

	y_h = grid_search.predict(X_test)

	print(f'accuracy: {accuracy_score(y_test, y_h)}')

	print(classification_report(y_test, y_h))


