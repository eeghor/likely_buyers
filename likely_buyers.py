from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.decomposition import PCA
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
import arrow
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer

class PrevActivityCounts(TransformerMixin):

	"""
	extract total previous bookings, quotes and cancellations for each customers and for every booking or quote
	"""

	def transform(self, X, **kwargs):

		X_ = X[['CustomerId', 'BookingId', 'CreatedOn', 'Cancelled', 'isBooking']].sort_values(['CustomerId', 'CreatedOn'])
		# X_['isBooking'] = X_['Type'].apply(lambda x: 1 if x == 'Booking' else 0)
		X_['isQuote'] = X_['isBooking'].apply(lambda x: 0 if x == 1 else 1)

		prevs = []
		

		for r in X_[['CustomerId', 'isBooking', 'isQuote', 'Cancelled']].groupby('CustomerId'):
			
			prevs.append(r[1].cumsum().shift(1, fill_value=0)[['isBooking', 'isQuote', 'Cancelled']] \
				 .rename(columns={'isBooking': 'prev_bks', 'isQuote': 'prev_qts', 'Cancelled': 'prev_cnl'}))

		return X.join(pd.concat(prevs), how='inner')[['prev_bks', 'prev_qts', 'prev_cnl']]

	def fit(self, X, y=None, **kwargs):
		return self

class VehicleType(TransformerMixin):

	"""
	extract total previous bookings for each customers (for every booking or transaction)
	"""

	def transform(self, X, **kwargs):

		return X[['isCar', 'is4x4', 'isCamper', 'isMinibus', 'isMotorHome']]

	def fit(self, X, y=None, **kwargs):
		return self

class TripDetails(TransformerMixin):

	"""
	extract total previous bookings for each customers (for every booking or transaction)
	"""

	def transform(self, X, **kwargs):

		return pd.concat([
				pd.get_dummies(X['ToCountry'], prefix='to'),
				pd.get_dummies(X['FromDayWeek'], prefix='from'),
				pd.get_dummies(X['ToDayWeek'], prefix='to'),
				X[['DurationDays', 'UpfrontDays', 'Cancelled']]], sort=False, axis=1)

	def fit(self, X, y=None, **kwargs):
		return self

class CustomerDetails(TransformerMixin):

	"""
	extract total previous bookings for each customers (for every booking or transaction)
	"""

	def transform(self, X, **kwargs):

		return pd.concat([
				pd.get_dummies(X['ResCountry'], prefix='rc'),
				pd.get_dummies(X['Lang'], prefix='ws_lang')], sort=False, axis=1)

	def fit(self, X, y=None, **kwargs):
		return self

class Payment(TransformerMixin):

	"""
	extract total previous bookings for each customers (for every booking or transaction)
	"""

	def transform(self, X, **kwargs):

		return X[['Paid', 'Coupon']]

	def fit(self, X, y=None, **kwargs):
		return self

class QuoteTiming(TransformerMixin):

	"""
	extract total previous bookings for each customers (for every booking or transaction)
	"""

	def transform(self, X, **kwargs):

		return pd.concat([
				pd.get_dummies(X['QuoteWeek'], prefix='qwk'),
				pd.get_dummies(X['QuoteDay'], prefix='qd'),
				pd.get_dummies(X['QuoteHour'], prefix='qh')], sort=False, axis=1)

	def fit(self, X, y=None, **kwargs):
		return self

class PreviousBookingsSameCountry(TransformerMixin):

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

				this_country = cr[1]['ToCountry']

				prev_acts = d.iloc[:i]

				prev_actv_this_country = prev_acts[prev_acts['ToCountry'] == this_country]

				if not prev_actv_this_country.empty:

					prev_bks_this_country = len(prev_actv_this_country[prev_actv_this_country['isBooking'] == 1])
					prev_qts_this_country = len(prev_actv_this_country[prev_actv_this_country['isBooking'] == 0])
					prev_cnl_this_country = len(prev_actv_this_country[prev_actv_this_country['Cancelled'] == 1])
				else:
					prev_bks_this_country = prev_qts_this_country = prev_cnl_this_country = 0

				prev_bks.append(prev_bks_this_country)
				prev_qts.append(prev_qts_this_country)
				prev_cnl.append(prev_cnl_this_country)

			prevs.append(pd.DataFrame({'prev_bks_this_country': prev_bks,
							'prev_qts_this_country': prev_qts,
							'prev_cnl_this_country': prev_cnl}).set_index(d.index))

		return pd.concat(prevs)

	def fit(self, X, y=None, **kwargs):
		return self

class PreviousBookingsThisWeek(TransformerMixin):

	"""
	extract total previous bookings for each customers (for every booking or transaction)
	"""

	def transform(self, X, **kwargs):

		prevs1 = [] 
		prevs2 = []

		for p, w in zip([prevs1, prevs2], ['QuoteWeek', 'QuoteDay']):

			for r in X.groupby('CustomerId'):

				d = r[1].sort_values('CreatedOn')

				prev_bks = []
				prev_qts = []
				prev_cnl = []
			
				for i, cr in enumerate(d.iterrows()):

					this_week = cr[1][w]

					prev_acts = d.iloc[:i]

					prev_actv_this_country = prev_acts[prev_acts[w] == this_week]

					if not prev_actv_this_country.empty:

						prev_bks_this_country = len(prev_actv_this_country[prev_actv_this_country['isBooking'] == 1])
						prev_qts_this_country = len(prev_actv_this_country[prev_actv_this_country['isBooking'] == 0])

					else:
						prev_bks_this_country = prev_qts_this_country = prev_cnl_this_country = 0

					prev_bks.append(prev_bks_this_country)
					prev_qts.append(prev_qts_this_country)
					prev_cnl.append(prev_cnl_this_country)

				p.append(pd.DataFrame({'prev_bks_this_' + w: prev_bks,
							'prev_qts_this_' + w: prev_qts,
							'prev_cnl_this_' + w: prev_cnl}).set_index(d.index))

		return pd.concat([pd.concat(prevs1), pd.concat(prevs2)], axis=1)

	def fit(self, X, y=None, **kwargs):
		return self

class PreviousBookingsSameDay(TransformerMixin):

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

				prev_acts = d.iloc[:i]

				prev_actv_this_country = prev_acts[prev_acts['CreatedOnDate'] == this_day]

				if not prev_actv_this_country.empty:

					prev_bks_this_country = len(prev_actv_this_country[prev_actv_this_country['isBooking'] == 1])
					prev_qts_this_country = len(prev_actv_this_country[prev_actv_this_country['isBooking'] == 0])

				else:
					prev_bks_this_country = prev_qts_this_country = prev_cnl_this_country = 0

				prev_bks.append(prev_bks_this_country)
				prev_qts.append(prev_qts_this_country)
				prev_cnl.append(prev_cnl_this_country)

			prevs.append(pd.DataFrame({'prev_bks_same_day': prev_bks,
							'prev_qts_same_day': prev_qts,
							'prev_cnl__same_day': prev_cnl}).set_index(d.index))

		return pd.concat(prevs)

	def fit(self, X, y=None, **kwargs):
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

class PropensityEstimator:

	def __init__(self):
		
		self.vars = {'cat': 'FromDayWeek ToDayWeek QuoteWeek QuoteDay QuoteHour Lang ResCountry'.split()}

	def load_data(self, tr_file='B2C_Rentalcover_31DEC2019.csv'):

		self.d = pd.read_csv('data/' + tr_file, 
					parse_dates=['FromDate','ToDate','CreatedOn', 'CreatedOnDate'])

		print(f'{self.d["CustomerId"].nunique():,} customer ids')
		print(f'{len(self.d[self.d["isBooking"] == 1]):,} bookings')
		print(f'{len(self.d[self.d["isBooking"] == 0]):,} quotes')

		# print(pd.get_dummies(self.d.iloc[:30][self.vars['cat']], prefix='is'))

		return self

if __name__ == '__main__':
	
	pe = PropensityEstimator().load_data()

	features = FeatureUnion([('prev_activity_counts', PrevActivityCounts()), 
							 ('vehicle_type', VehicleType()),
							 ('trip_details', TripDetails()), 
							 ('customer_details', CustomerDetails()), 
							 ('payment', Payment()), 
							 ('quote_timing', QuoteTiming()), 
							 ('prev_activities_same_country', PreviousBookingsSameCountry()), 
							 ('prev_bks_this_week', PreviousBookingsThisWeek()), 
							 ('prev_bks_this_day', PreviousBookingsSameDay()), 
							 ('prev_qts_this_trip', PrevQtsThisTrip())])

	pd.DataFrame(features.fit_transform(pe.d)).to_csv('fts.csv', index=False)

