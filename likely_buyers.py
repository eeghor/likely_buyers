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

		tr_details_cat = 'ToCountry FromDayWeek ToDayWeek'.split()
		tr_details_asis = 'DurationDays UpfrontDays Cancelled'.split()

		return pd.concat([pd.get_dummies(X[c], prefix=c.lower()) for c in tr_details_cat] + [X[tr_details_asis]],
			sort=False, axis=1)

	def fit(self, X, y=None):
		return self

class CustomerDetails(TransformerMixin):

	"""
	basic customer details
	"""

	def transform(self, X):

		cus_details_cols = 'ResCountry Lang'.split()

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
       'FirstName', 'LastName', 'Email', 'ResCountry', 'prev_bks', 'prev_qts',
       'prev_cnl', 'prev_act_bk', 'fst_act_bk', 'last_act_same_cnt',
       'prev_act_same_cnt', 'prev_diff_cnt'
		"""
		
		self.cols_to_parse_date = 'FromDate ToDate CreatedOn CreatedOnDate'.split()
		self.cols_to_drop = 'CustomerId BookingId Reference'.split()

	def load(self, file='B2C_Rentalcover_31DEC2019.csv'):

		self.data = pd.read_csv('data/' + file, parse_dates=self.cols_to_parse_date)

		print(f'{len(self.data):,} rows')
		print(f'{self.data["CustomerId"].nunique():,} customer ids')
		print(f'{self.data["Reference"].nunique():,} references')
		print(f'{len(self.data[self.data["isBooking"] == 1]):,} bookings')
		print(f'{len(self.data[self.data["isBooking"] == 0]):,} quotes')

		self.data = self.data.drop(self.cols_to_drop, axis=1)

		cs = ['AU', 'US', 'IT', 'GB', 'ES', 'NZ', 'CA', 'FR', 'DE', 'PT', 'GR', 'IS',
		       'JP', 'IE', 'CH', 'ZA', 'NL', 'AT', 'MX', 'HR']

		self.data['ToCountry'] = self.data['ToCountry'].apply(lambda x: x if x in cs else 'XX')
		self.data['ResCountry'] = self.data['ResCountry'].apply(lambda x: x if x in cs else 'XX')


		return self

if __name__ == '__main__':
	
	dl = DataLoader().load()

	X = dl.data[[c for c in dl.data.columns if c != 'isBooking']]
	y = dl.data['isBooking'].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=278)

	print(f'bookings in training/test set {sum(y_train):,}/{sum(y_test):,}')

	features = FeatureUnion([('trip_details', TripDetails()), 
							 ('vehicle_type', VehicleType()), 
							 ('cust_details', CustomerDetails()), 
							 ('payment_details', PaymentDetails()), 
							 ('quote_timing', QuoteTiming())])

	pipe = make_pipeline(features, StandardScaler(), RandomForestClassifier())

	pipe.fit(X_train, y_train)

	y_h = pipe.predict(X_test)

	print(f'accuracy: {accuracy_score(y_test, y_h)}')

	print(classification_report(y_test, y_h))


