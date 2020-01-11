from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
import arrow
import numpy as np
import json

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer

outbound_uk = pd.read_csv('data/outbound_tourism_uk.csv')
outbound_uk['share'] = round(outbound_uk['2018']/outbound_uk['2018'].sum(),4)
outbound = defaultdict(lambda: defaultdict())

for country, share in zip(outbound_uk['iso_code'].values, outbound_uk['share']):
	outbound['uk'][country] = share

cnt_lefthand = pd.read_csv('data/countries_lefthand.csv')

competitor_prices = {'AU': json.load(open('data/prices_pday_au.json'))}

cheapest_pday = {'AU': np.amin(np.vstack((np.array(competitor_prices['AU'][comp].get('0_pday')) 
						for comp in competitor_prices['AU'] if comp != 'rentalcover')), axis=0)}


# popular touristic destinations
touristic_dests = {'UK': ['ES', 'FR', 'IT', 'US', 'IE', 'PT', 'DE', 'NL', 'PL', 'GR'],
					'AU': ['ID', 'NZ', 'US', 'TH', 'IN', 'CN', 'UK', 'SG', 'JP', 'MY', 'HK', 'FJ', 'KR'],
					'NZ': ['AU', 'US', 'FJ', 'UK', 'CN'],
					'US': ['IN', 'JP', 'CN', 'TW', 'TH', 'PH', 'BH', 'DO', 'JA', 'CR', 'CZ', 'HU', 'PL',
							'AT', 'FR', 'DE', 'IE', 'IS', 'IT', 'NL', 'ES', 'UK', 'IL', 'AU', 'CO'],
							'RU': ['TR', 'DE', 'TH', 'IT', 'ES', 'AE', 'CY', 'GR', 'TN', 'VN','FR',
							'CZ', 'IL', 'ME', 'AT', 'NL', 'US']}

def competitor_quote(rental_company, days, cheapest=True):

	_rental_company = rental_company.lower().strip()

	if _rental_company == 'thrifty':

		if days < 4:
			perday = 8.81*cheapest + 15.0*(1.0 - cheapest)
		elif 4 <= days < 14:
			perday = 7.15*cheapest + 11.5*(1.0 - cheapest)
		else:
			perday = 6.05*cheapest + 9.05*(1.0 - cheapest)

	elif _rental_company == 'europcar':

		if days == 1:
			perday = 23.0
		elif 2 <= days <= 3:
			perday = 20.5
		elif 4 <= days <= 6:
			perday = 18.0
		elif days in (set(range(7, 14)) - {8}):
			perday = 13.0
		elif days == 8:
			perday = 17.30
		elif 14 <= days <= 27:
			perday = 10.5
		else:
			perday = 8.0

	elif _rental_company == 'herz':

		perday = 26.40

	elif _rental_company == 'sixt':

		if days < 6:
			perday = 13.0*cheapest + 28.5*(1.0 - cheapest)
		elif days == 6:
			perday = 10.99*cheapest + 24.5*(1.0 - cheapest)
		elif days == 7:
			perday = 10.49*cheapest + 23.5*(1.0 - cheapest)
		elif 8 <= days <= 14:
			perday = 10.0*cheapest + 21.40*(1.0 - cheapest)
		else:
			perday = 10.0*cheapest + 20.30*(1.0 - cheapest)

	return days*perday

class ModelTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return self.model.predict(X)

class VehicleType(BaseEstimator, TransformerMixin):

	"""
	what sort of a vehicle a customer was interested in
	"""

	def transform(self, X):

		veh_type_cols = 'isCar is4x4 isCamper isMinibus isMotorHome'.split()

		return X[veh_type_cols]

	def fit(self, X, y=None):
		return self

class PotentialSavings(BaseEstimator, TransformerMixin):

	"""
	potential savings if a customer decides to accept the quoted price compared to the lowest
	competitor price
	"""

	def transform(self, X):

		savings = []

		for i, (cnt, ndays, usd_paid) in enumerate(zip(X['ToCountry'], X['DurationDays'], X['Paid'])):
			if (cnt in cheapest_pday):
				use_price_perday = cheapest_pday[cnt][ndays] if ndays < 21 else cheapest_pday[cnt][-1]
				savings.append(use_price_perday*ndays*0.69 - usd_paid)
			else:
				savings.append(0)


		return pd.DataFrame({'savings': savings})

	def fit(self, X, y=None):
		return self

class TripDetails(BaseEstimator, TransformerMixin):

	"""
	available trip details
	"""

	def transform(self, X):


		# tr_details_cat = 'FromDayWeek ToDayWeek'.split()
		# tr_details_asis = 'DurationDays UpfrontDays Cancelled'.split()

		# return X[tr_details_asis]
		return X

	def fit(self, X, y=None):
		return self

class PrevActivities(BaseEstimator, TransformerMixin):

	"""
	available trip details
	"""

	def transform(self, X):

		# pa_cols = 'prev_bks prev_qts prev_cnl prev_act_bk fst_act_bk last_act_same_cnt prev_act_same_cnt prev_diff_cnt'.split()

		return X

	def fit(self, X, y=None):
		return self

class ToFromCountries(BaseEstimator, TransformerMixin):

	"""
	where is a customer renting a car and what's his country or residence
	"""

	def transform(self, X):

		top30_to = ['AU', 'US', 'IT', 'GB', 'ES', 'CA', 'NZ', 'DE', 'FR', 'IS', 'PT', 
					'GR', 'JP', 'IE', 'CH', 'NL', 'ZA', 'MX', 'AT', 'TH', 'PL', 'HR', 
					'CY', 'IL', 'NO', 'RO', 'BE', 'TW', 'CZ', 'HU']

		to_country_names = ['tocountry_' + c for c in top30_to] + ['tocountry_XX']

		top30_from = ['AU', 'GB', 'RU', 'US', 'CA', 'IL', 'TW', 'ES', 'HK', 'SG', 'IT',
					  'FR', 'DE', 'CH', 'NZ', 'NL', 'JP', 'KR', 'UA', 'PL', 'MY', 'AE', 
					  'BE', 'IE', 'CN', 'BR', 'AR', 'TH', 'IN', 'CZ']

		from_country_names = ['rescountry_' + c for c in top30_from] + ['rescountry_XX']

		cs = {'C1': ['AU', 'NZ', 'US', 'GB', 'CA', 'IE'],
			  'C2': ['IT', 'ES', 'FR', 'PT', 'GR', 'CY', 'HR', 'ME', 'TR'],
			  'C3': ['UA', 'RU', 'MD', 'BY', 'PL', 'CZ', 'HU'],
			  'C4': ['DE', 'CH', 'NL', 'SE', 'AT', 'FI', 'NO', 'DK']}

		_Y1 = X['ToCountry'].apply(lambda x: x if x in top30_to else 'XX')
		_Y2 = X['ResCountry'].apply(lambda x: x if x in top30_from else 'XX')

		_X1 = pd.get_dummies(_Y1, prefix='tocountry')
		_X2 = pd.get_dummies(_Y2, prefix='rescountry')

		for n in to_country_names:
			if n not in _X1.columns:
				_X1[n] = 0

		for n in from_country_names:
			if n not in _X2.columns:
				_X2[n] = 0

		_X3 = pd.DataFrame({'tocountry_lh': X['ToCountry'].isin(cnt_lefthand['iso_code'])*1})
		_X4 = pd.DataFrame({'rescountry_lh': X['ResCountry'].isin(cnt_lefthand['iso_code'])*1})

		# cnt_cols = 'ToCountry ResCountry'.split()

		# _X = pd.concat([pd.get_dummies(X[c], prefix=c.lower()) for c in cnt_cols], sort=False, axis=1)
		# _X['US_tour_dest'] = X['ToCountry'].apply(lambda x: 1 if x in touristic_dests['US'] else 0)
		# _X['AU_tour_dest'] = X['ToCountry'].apply(lambda x: 1 if x in touristic_dests['AU'] else 0)
		# _X5 = _Y1.apply(lambda x: 1 if x in touristic_dests['UK'] else 0)
		_X5 = pd.DataFrame({'pop_dest_uk': _Y1.apply(lambda x: outbound['UK'][x] if x in outbound['UK'] else 0)})
		# _X['from_lh_country'] = X['ResCountry'].isin(cnt_lefthand['iso_code'])*1
		# _X['to_lh_country'] = X['ToCountry'].isin(cnt_lefthand['iso_code'])*1
		# _X['NZ_tour_dest'] = X['ToCountry'].apply(lambda x: 1 if x in touristic_dests['NZ'] else 0)
		# _X['RU_tour_dest'] = X['ToCountry'].apply(lambda x: 1 if x in touristic_dests['RU'] else 0)

		# all_dummy_names = sorted(['_'.join([cl.lower(), c]) for c in 'C1 C2 C3 C4 XX'.split() for cl in cnt_cols])

		# for c in all_dummy_names:
		# 	if c not in _X.columns:
		# 		_X[c] = 0

		return pd.concat([_X1, _X2, _X3, _X4, _X5], axis=1, sort=False)


	def fit(self, X, y=None):
		return self

class CustomerDetails(BaseEstimator, TransformerMixin):

	"""
	basic customer details
	"""

	def transform(self, X):

		lang_names = sorted(['lang_' + l for l in 'en de fr es ru jp xx'.split()])

		s = pd.get_dummies(X['Lang'].apply(lambda x: x if x in 'en de fr es ru jp'.split() else 'xx'), prefix='lang')

		for nm in lang_names:
			if nm not in s.columns:
				s[nm] = 0

		return s[lang_names]

	def fit(self, X, y=None):
		return self

class PaymentDetails(BaseEstimator, TransformerMixin):

	"""
	potential or actual payment related: how much in USD was the quote, was there the permanent coupon
	"""

	def transform(self, X):

		paym_cols = 'Paid Coupon'.split()

		return X[paym_cols]

	def fit(self, X, y=None):
		return self

class QuoteTiming(BaseEstimator, TransformerMixin):

	"""
	mostly time of the quote or booking
	"""

	def transform(self, X):

		qmonth_names = ['quotemonth_' + m for m in sorted('Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split())]
		qweek_names = ['quoteweek_' + str(c) for c in range(1,53)]
		qday_names = ['quoteday_' + d for d in 'Sun Mon Tue Wed Thu Fri Sat'.split()]
		qhr_names = [f'quotehour_{h:02}' for h in range(25)]

		dummy_months = pd.get_dummies(X['QuoteMonth'], prefix='quotemonth')
		dummy_weeks = pd.get_dummies(X['QuoteWeek'], prefix='quoteweek')
		dummy_days = pd.get_dummies(X['QuoteDay'], prefix='quoteday')
		# dummy_hrs = pd.get_dummies(X['QuoteHour'])

		for w in qmonth_names:
			if w not in dummy_months.columns:
				dummy_months[w] = 0

		dummy_months = dummy_months[qmonth_names]

		for w in qweek_names:
			if w not in dummy_weeks.columns:
				dummy_weeks[w] = 0

		dummy_weeks = dummy_weeks[qweek_names]

		for w in qday_names:
			if w not in dummy_days.columns:
				dummy_days[w] = 0

		dummy_days = dummy_days[qday_names]

		# for w in qhr_names:
		# 	if w not in dummy_hrs.columns:
		# 		dummy_hrs[w] = 0

		# dummy_hrs = dummy_hrs[qhr_names]

		return pd.concat([dummy_months, dummy_weeks], sort=False, axis=1)

	def fit(self, X, y=None):
		return self

class PrevQtsThisTrip(BaseEstimator, TransformerMixin):

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
		"""
		
		self.cols_to_parse_date = 'FromDate ToDate CreatedOn CreatedOnDate'.split()
		self.cols_to_drop = 'CustomerId BookingId Reference'.split()

	def load(self, file='B2C_Rentalcover_08JAN2020.csv'):

		self.data = pd.read_csv('data/' + file, parse_dates=self.cols_to_parse_date)

		print(f'{len(self.data):,} rows')
		print(f'{self.data["CustomerId"].nunique():,} customer ids')
		print(f'{self.data["Reference"].nunique():,} references')
		print(f'{len(self.data[self.data["isBooking"] == 1]):,} bookings')
		print(f'{len(self.data[self.data["isBooking"] == 0]):,} quotes')

		self.data = self.data.drop(self.cols_to_drop, axis=1).fillna(0)

		return self

if __name__ == '__main__':
	
	dl = DataLoader().load()

	X = dl.data[[c for c in dl.data.columns if c != 'isBooking']]
	y = dl.data['isBooking'].values

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=278)

	print(f'bookings in training/test set {sum(y_train):,}/{sum(y_test):,}')

	pipe = Pipeline([('features', FeatureUnion([
										 
										 ('ct', ColumnTransformer([('trip_details', 
											 								   TripDetails(), 
											 								  ['DurationDays', 'UpfrontDays', 'Cancelled']),
										 ('prev_activities', PrevActivities(), 'prev_bks prev_qts prev_cnl prev_act_bk fst_act_bk last_act_same_cnt prev_act_same_cnt prev_diff_cnt'.split())]))
							 			 # ('to_from_countries', ToFromCountries()), 
							 			 # ('prev_activities', PrevActivities()),
							 			 # ('vehicle_type', VehicleType()), 
							 			 # ('cust_details', CustomerDetails()), 
							 			 # ('payment_details', PaymentDetails()), 
							 			 # ('quote_timing', QuoteTiming()), 
							 			 # ('potential_savings', PotentialSavings())
							 			 ])
					 ),
			   		('classifiers', FeatureUnion([('randomforest', RandomForestClassifier()),
			   									  ('gradboosting', GradientBoostingClassifier()),
			   									  ('kmeans', ModelTransformer(KMeans(n_clusters=2))),
			   									  ('adaboost', AdaBoostClassifier())])),
			   		('last_classifier', KNeighborsClassifier(n_neighbors=3))
			   		])

	pipe = make_pipeline(features, StandardScaler(), RandomForestClassifier())

	pars = {'classifiers__randomforest__n_estimators': (50, 100, 150, 200, 300)}

	# grid_search = GridSearchCV(pp, pars, n_jobs=2, verbose=1, cv=4)

	grid_search = GridSearchCV(pipe, pars, n_jobs=2, verbose=1, cv=4)

	grid_search.fit(X_train, y_train)

	y_h = grid_search.predict(X_test)

	print(f'accuracy: {accuracy_score(y_test, y_h):06.4f}')

	print(classification_report(y_test, y_h))

	# print(confusion_matrix(y_test, y_h))


