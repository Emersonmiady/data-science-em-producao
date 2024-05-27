import math
import pickle
import datetime
import inflection
import numpy as np
import pandas as pd

class Rossmann(object):
    '''
    Initializes the Rossmann class.

    This function initializes the Rossmann class and loads the scalers for the respective parameters.
    The scalers are loaded from pickle files located in the '../parameter/' directory.

    Parameters:
        None

    Returns:
        None
    '''
    def __init__(self):
        # Parameter path
        self.param_path = '../parameter/'

        # Scalers
        self.year_scaler = pickle.load(open(self.param_path + 'year_scaler.pkl', 'rb'))
        self.promo_time_week_scaler = pickle.load(open(self.param_path + 'promo_time_week_scaler.pkl', 'rb'))
        self.competition_distance_scaler = pickle.load(open(self.param_path + 'competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.param_path + 'competition_time_month_scaler.pkl', 'rb'))
        self.store_type_scaler = pickle.load(open(self.param_path + 'store_type_scaler.pkl', 'rb'))
        
    def change_cols_to_snake_case(self, df):
        '''
        Change a list of columns of dataframe from 
        camel case to snake case.
        This function returns the modified dataframe.
        '''
        cols = df.columns
        snake_case = lambda x: inflection.underscore(x)
        new_cols = list(map(snake_case, cols))
        df.columns = new_cols
        
        return df

    def fillout_nan_on_col_with(self, time, col, df):
        '''
        According to "date" column on dataframe, this function fills NaN on certain attribute with
        the wanted time type. The time should be "year", "month" or "week".
        The function returns the modified dataframe.
        '''
        if time == 'year':
            df[col] = df.apply(lambda x: x['date'].year if math.isnan(x[col]) else x[col], axis=1)
            return df
        elif time == 'month':
            df[col] = df.apply(lambda x: x['date'].month if math.isnan(x[col]) else x[col], axis=1)
            return df
        elif time == 'week':
            df[col] = df.apply(lambda x: x['date'].week if math.isnan(x[col]) else x[col], axis=1)
            return df
        else:
            print('Unrecognized time...')
            return None

    def cyclical_transformation(self, variable, df, period):
        '''
        Transforming the respective period variable that has a cycle in sine and cossine.
        Return the new dataframe.
        '''
        df[f'{variable}_sin'] = df[variable].apply(lambda x: np.sin(x * (2 * np.pi / period)))
        df[f'{variable}_cos'] = df[variable].apply(lambda x: np.cos(x * (2 * np.pi / period)))

        df = df.drop(variable, axis=1)
        return df

        
    def data_cleaning(self, df):
        '''
        All data cleaning for model application.
        '''
        
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 
                    'Promo2SinceYear', 'PromoInterval']

        df = df[cols_old]

        # Renaming data columns to snake_case
        df = self.change_cols_to_snake_case(df)
        
        # Changing the date type to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Fill NaN on competition_distance column
        max_value = df['competition_distance'].max()
        df['competition_distance'] = df['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)
        
        # Fill NaN on time columns
        df = self.fillout_nan_on_col_with('month', 'competition_open_since_month', df)
        df = self.fillout_nan_on_col_with('year', 'competition_open_since_year', df)
        df = self.fillout_nan_on_col_with('week', 'promo2_since_week', df)
        df = self.fillout_nan_on_col_with('year', 'promo2_since_year', df)
        
        # Fill NaN on promo_interval column
        df['promo_interval'].fillna(0, inplace=True)
        
        # Mapping the current number of the month to 3 first letter of the month
        month_map = {1 : 'Jan', 2 : 'Feb', 3 : 'Mar', 4 : 'Apr',
                     5 : 'May', 6 : 'Jun', 7 : 'Jul', 8 : 'Aug',
                     9 : 'Sep', 10 : 'Oct', 11 : 'Nov', 12 : 'Dec'}
        
        # month_map
        df['month_map'] = df['date'].dt.month.map(month_map)
        
        # is_promo
        df['is_promo'] = df.apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)
        
        # Changing some time columns
        wanted_cols = ['competition_open_since_month', 
                       'competition_open_since_year', 
                       'promo2_since_week', 
                       'promo2_since_year']

        for col in wanted_cols:
            df[col] = df[col].astype('int64')
            
        return df
    
    def feature_engineering(self, df):
        '''
        All feature engineering for model application.
        '''
        
        # Year
        df['year'] = df['date'].dt.year

        # Month
        df['month'] = df['date'].dt.month

        # Day
        df['day'] = df['date'].dt.day

        # Week of year
        df['week_of_year'] = df['date'].dt.weekofyear

        # Year week
        df['year_week'] = df['date'].dt.strftime('%Y-%W')

        # Before 16th variable
        df['before_16th'] = df['day'].apply(lambda x: 'Yes' if x < 16 else 'No')
        
        # Competition since ("Year-Month-01" format)
        df['competition_since'] = df.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], 
                                          month=x['competition_open_since_month'], 
                                          day=1), axis=1)

        # Competition time month
        
        ## First we catch the number of months between the competition_since 
        ## and the "current" date
        ## Next we apply the dt.days method to catch the "number of days", 
        ## i.e. the number of months
        ## Finally we transform that datetime number in "int64"
        
        df['competition_time_month'] = ((df['date'] - df['competition_since'])/30)\
                                        .apply(lambda x: x.days).astype('int64')

        # Promo since ("Year-Month-1" format)
        df['promo_since'] = df['promo2_since_year'].astype(str) + '-' + df['promo2_since_week']\
                            .astype(str) # str transformation
        
        # datetime transformation
        df['promo_since'] = df['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))

        # Promo time week (analogous to competition_time_month creation)
        df['promo_time_week'] = ((df['date'] - df['promo_since'])/7).apply(lambda x: x.days).astype('int64')
        
        # Renaming the columns below
        
        ## Assortment (a = basic | b = extra | c = extended)
        df['assortment'] = df['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        ## State Holiday (a = Public holiday | b = Easter holiday | c = Christmas | 0 = Regular day)
        df['state_holiday'] = df['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')
        
        # Filtering open
        df = df[df['open'] != 0]
        
        # Fitering open, customers, promo_interval and month_map columns
        wanted_cols = ['open', 'promo_interval', 'month_map']
        df = df.drop(wanted_cols, axis=1)
        
        return df
    
    def data_preparation(self, df):
        '''
        All data preparation for model application.
        '''
        
        # year scaled
        df['year'] = self.year_scaler.fit_transform(df[['year']].values)
        
        # promo_time_week scaled
        df['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df[['promo_time_week']].values)
        
        # promo_time_week scaled
        df['competition_distance'] = self.competition_distance_scaler.fit_transform(df[['competition_distance']].values)
        
        # competition_time_month scaled
        df['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df[['competition_time_month']].values)
        
        # state_holiday - One Hot Encoding
        df = pd.get_dummies(df, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding
        df['store_type'] = self.store_type_scaler.fit_transform(df['store_type'])

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1,
                           'extra': 2,
                           'extended': 3}

        df['assortment'] = df['assortment'].map(assortment_dict)
        
        # Cyclical transformation
        
        # day_of_week
        df = self.cyclical_transformation('day_of_week', df, 7)

        # month
        df = self.cyclical_transformation('month', df, 12)

        # day
        df = self.cyclical_transformation('day', df, 30)

        # week_of_year
        df = self.cyclical_transformation('week_of_year', df, 52)
        
        # Droping some cols
        cols_drop = ['promo_since', 'competition_since', 'year_week', 'before_16th']
        df = df.drop(cols_drop, axis=1)
        
        # Columns selected by boruta
        cols_selected = ['store', 'promo', 'store_type', 'assortment', 
                         'competition_distance', 'competition_open_since_month', 
                         'competition_open_since_year', 'promo2', 
                         'promo2_since_week', 'promo2_since_year', 
                         'competition_time_month', 'promo_time_week', 
                         'day_of_week_sin', 'day_of_week_cos', 'month_sin', 
                         'month_cos', 'day_sin', 'day_cos', 
                         'week_of_year_sin', 'week_of_year_cos']
        
        return df[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        '''
        After data preparation, get the prediction for the test data.
        '''

        # Prediction
        pred = model.predict(test_data)

        # Join pred into original data
        original_data['Prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')