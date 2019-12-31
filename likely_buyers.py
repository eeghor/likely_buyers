from sklearn.pipeline import Pipeline
import pandas as pd
from collections import Counter, defaultdict
import arrow
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer

"""
asis_features = ['QuoteWeek', 'QuoteDay', 'QuoteHour', 'Paid', 'Coupon', , 'Lang', 'ResCountry']
"""
class PreviousBookings(TransformerMixin):

	"""
	extract total previous bookings for each customers (for every booking or transaction)
	"""

	def transform(self, X, **kwargs):

		X_ = X[['CustomerId', 'Type', 'BookingId', 'CreatedOn', 'Cancelled']].sort_values(['CustomerId', 'CreatedOn'])
		X_['isBooking'] = X_['Type'].apply(lambda x: 1 if x == 'Booking' else 0)
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

					prev_bks_this_country = len(prev_actv_this_country[prev_actv_this_country['Type'] == 'Booking'])
					prev_qts_this_country = len(prev_actv_this_country[prev_actv_this_country['Type'] == 'Quote'])
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

		prevs = [] 

		w = 'QuoteWeek' if kwargs['what'] == 'week' else 'QuoteDay'

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

					prev_bks_this_country = len(prev_actv_this_country[prev_actv_this_country['Type'] == 'Booking'])
					prev_qts_this_country = len(prev_actv_this_country[prev_actv_this_country['Type'] == 'Quote'])

				else:
					prev_bks_this_country = prev_qts_this_country = prev_cnl_this_country = 0

				prev_bks.append(prev_bks_this_country)
				prev_qts.append(prev_qts_this_country)
				prev_cnl.append(prev_cnl_this_country)

			prevs.append(pd.DataFrame({'prev_bks_this_' + kwargs['what']: prev_bks,
							'prev_qts_this_' + kwargs['what']: prev_qts,
							'prev_cnl_this_' + kwargs['what']: prev_cnl}).set_index(d.index))

		return pd.concat(prevs)

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

					prev_bks_this_country = len(prev_actv_this_country[prev_actv_this_country['Type'] == 'Booking'])
					prev_qts_this_country = len(prev_actv_this_country[prev_actv_this_country['Type'] == 'Quote'])

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

class PreviousBookingsSameTrip(TransformerMixin):

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
															for c in prev_actv_this_country[prev_actv_this_country['Type'] == 'Quote']['FromDate']]),
												sum([arrow.get(c).shift(days=-2) <= arrow.get(this_day) <= arrow.get(c).shift(days=+2) 
															for c in prev_actv_this_country[prev_actv_this_country['Type'] == 'Quote']['ToDate']]))

				else:
					prev_bks_this_country = 0

				prev_bks.append(prev_bks_this_country)

			prevs.append(pd.DataFrame({'prev_qts_same_trip': prev_bks}).set_index(d.index))

		return pd.concat(prevs)

	def fit(self, X, y=None, **kwargs):
		return self

class CatFeatures(TransformerMixin):

	"""
	extract total previous bookings for each customers (for every booking or transaction)
	"""

	def transform(self, X, **kwargs):

		fs = 'FromDayWeek ToDayWeek QuoteWeek QuoteDay QuoteHour Lang ResCountry'.split()

		return pd.get_dummies(X[fs], prefix='is')

	def fit(self, X, y=None, **kwargs):
		return self

class PropensityEstimator:

	def __init__(self):
		
		self.vars = {'cat': 'FromDayWeek ToDayWeek QuoteWeek QuoteDay QuoteHour Lang ResCountry'.split()}

	def load_data(self, tr_file='B2C_Rentalcover_24DEC2019.csv'):

		self.d = pd.read_csv('data/' + tr_file, 
					parse_dates=['FromDate','ToDate','CreatedOn', 'CreatedOnDate'])

		# print(pd.get_dummies(self.d.iloc[:30][self.vars['cat']], prefix='is'))

		return self

if __name__ == '__main__':
	
	pe = PropensityEstimator().load_data()

	pl = TripDetails().transform(pe.d)

	print(pl)

