from sklearn.pipeline import Pipeline
import pandas as pd
from collections import Counter, defaultdict
import arrow
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer

class PreviousBookings(TransformerMixin):

	"""
	extract total previous bookings for each customers (for every booking or transaction)
	"""

	def transform(self, X, **kwargs):

		X_ = X[['CustomerId', 'Type', 'BookingId', 'CreatedOn']].sort_values(['CustomerId', 'CreatedOn'])
		X_['isBooking'] = X_['Type'].apply(lambda x: 1 if x == 'Booking' else 0)
		X_['isQuote'] = X_['isBooking'].apply(lambda x: 0 if x == 1 else 1)

		prevs = []

		for r in X_[['CustomerId', 'isBooking', 'isQuote']].groupby('CustomerId'):
			
			prevs.append(r[1].cumsum().shift(1, fill_value=0)[['isBooking', 'isQuote']] \
				 .rename(columns={'isBooking': 'prev_bks', 'isQuote': 'prev_qts'}))

		return pd.concat(prevs)

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

	pl = PreviousBookings().transform(pe.d)

	print(pl)

