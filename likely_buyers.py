from sklearn.pipeline import Pipeline
import pandas as pd
from collections import Counter, defaultdict
import arrow
import numpy as np

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer

class PreviousBookings(TransformerMixin):

    def transform(self, X, **transform_params):
        X.sort_values(['CustomerId', 'CreatedOn'])
        return hours

    def fit(self, X, y=None, **fit_params):
        return self

class PropensityEstimator:

	def __init__(self):
		
		self.vars = {'cat': 'FromDayWeek ToDayWeek QuoteWeek QuoteDay QuoteHour Lang ResCountry'.split()}

	def load_data(self, tr_file='B2C_Rentalcover_24DEC2019.csv'):

		self.d = pd.read_csv('data/' + tr_file, 
					parse_dates=['FromDate','ToDate','CreatedOn', 'CreatedOnDate'])

		print(pd.get_dummies(self.d.iloc[:30][self.vars['cat']], prefix='is'))

		return self

if __name__ == '__main__':
	
	pe = PropensityEstimator().load_data()

	print(pe.d) 

