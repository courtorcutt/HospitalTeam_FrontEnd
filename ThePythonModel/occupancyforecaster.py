import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.serialize import model_to_json, model_from_json
import matplotlib.pyplot as plt
import numpy as np
import json
import itertools


class OccupancyForecaster:
    """
    A class for forecasting hospital bed occupancy. Makes predictions by taking into account trends, seasonality, and holidays.
    """
    def __init__(self):
        self._model = None
        # Load the model
        try:
            with open('model.json', 'r') as fin:
                self._model = model_from_json(json.load(fin))
        except:
            print("Model file not found.")

    def forecast_occupancy(self, start_date, end_date):
        """
        Called by the GUI to receive the predicted occupancy for a given range of dates.
        Input dates should be strings in the format YYYY-MM-DD.
        :param start_date: <string> Start date for the forecast range
        :param end_date: <string> End date for the forecast range
        :return: <ndarray> Integer array of predicted occupancies
        """
        # Generate a range of dates as a dataframe object
        df = pd.DataFrame({'ds': pd.date_range(start=start_date, end=end_date)})

        # Predict occupancy for the range of dates
        forecast = self._model.predict(df)

        # Format the results
        forecast = np.array(forecast[['yhat']].values, dtype=int).flatten()

        return forecast

    def train_model(self, file_path):
        """
        Called by the GUI to enable the user to train the model with a given csv file.
        Input csv file should have a column named "date" with dates formatted as YYYY-MM-DD and
        a column named "occupancy" with numerical values of the bed occupancy.
        :param file_path: <string> File path for the training data
        :return: None
        """
        # LOAD DATA
        df = pd.read_csv(file_path)

        # PREPROCESS DATA
        data = df[['date', 'occupancy']]
        data.columns = ['ds', 'y']

        # TRAIN MODEL
        self._model = Prophet(changepoint_prior_scale=0.01, seasonality_prior_scale=5,
                              daily_seasonality=False, uncertainty_samples=0, yearly_seasonality=10)
        self._model.add_country_holidays(country_name='CA')
        self._model.fit(data)

        # SAVE MODEL
        with open('model.json', 'w') as fout:
            json.dump(model_to_json(self._model), fout)

    @staticmethod
    def _mape(predicted, actual):
        """
        Calculates Mean Absolute Percentage Error.
        :param predicted: Array of predicted occupancies
        :param actual: Array of actual occupancies
        :return: MAPE
        """
        return np.mean(np.abs((actual - predicted)/actual)) * 100

    @staticmethod
    def _rmse(predicted, actual):
        """
        Calculates Root Mean Squared Error.
        :param predicted: Array of predicted occupancies
        :param actual: Array of actual occupancies
        :return: RMSE
        """
        return np.sqrt(np.mean((predicted-actual)**2))

    @staticmethod
    def tune_hyperparameters():
        """
        Applies cross-validation to find the best hyperparameters for the model. (Takes a long time)
        :return: None
        """
        # LOAD DATA
        df = pd.read_csv("data.csv")

        # PREPROCESS DATA
        data = df[['date', 'occupancy']]
        data.columns = ['ds', 'y']

        # CROSS VALIDATION
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here
        size = len(all_params)
        counter = 0

        # Use cross validation to evaluate all parameters
        for params in all_params:
            m = Prophet(**params)
            m.add_country_holidays(country_name='CA')
            m.fit(data)
            df_cv = cross_validation(m, horizon='365 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])
            counter += 1
            print("PROGRESS:", counter, "/", size)

        # Print tuning results
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        print(tuning_results)

        # Print best result
        best_params = all_params[np.argmin(rmses)]
        print("Best Result:", best_params)

    @staticmethod
    def test_model():
        """
        Displays graphs of the model's performance on historical and new data.
        :return: None
        """
        # LOAD DATA
        df = pd.read_csv("data.csv")

        # PREPROCESS DATA
        data = df[['date', 'occupancy']]
        data.columns = ['ds', 'y']

        # Training data from 2014-04-21 to 2018-12-31 (4 years)
        train_data = df.loc[:1715, ['date', 'occupancy']]
        train_data.columns = ['ds', 'y']

        # Testing data from 2019-01-01 to 2019-03-31 (3 months)
        test_data = df.loc[1715:, ['date', 'occupancy']]
        test_data.columns = ['ds', 'y']
        test_data.reset_index(drop=True, inplace=True)

        # TEST MODEL
        model = Prophet(changepoint_prior_scale=0.01, seasonality_prior_scale=5,
                        daily_seasonality=False, uncertainty_samples=0, yearly_seasonality=10)
        model.add_country_holidays(country_name='CA')
        model.fit(data)

        df_cv = cross_validation(model, horizon='365 days', parallel="processes")
        y_predicted = np.array(df_cv[['yhat']].values).flatten()
        y_actual = np.array(df_cv[['y']].values).flatten()
        print("MAPE:", OccupancyForecaster._mape(y_predicted, y_actual))
        print("RMSE:", OccupancyForecaster._rmse(y_predicted, y_actual))
        df_p = performance_metrics(df_cv)
        print(df_p.head())
        plot_cross_validation_metric(df_cv, metric='mape')
        plt.show()
        plot_cross_validation_metric(df_cv, metric='rmse')
        plt.show()
        forecast = model.predict(data[['ds']])
        model.plot_components(forecast)

        model = Prophet(changepoint_prior_scale=0.01, seasonality_prior_scale=5,
                        daily_seasonality=False, uncertainty_samples=0, yearly_seasonality=10)
        model.add_country_holidays(country_name='CA')
        model.fit(train_data)

        forecast = model.predict(train_data[['ds']])
        forecast = forecast[['ds', 'yhat']]
        forecast.insert(2, "y", train_data['y'], True)
        forecast.columns = ['Dates', 'Predicted', 'Actual']
        forecast.plot(x='Dates')
        plt.ylabel("Occupancy")
        plt.ylim([0, 150])
        plt.title("4 year In-Sample Forecast")
        plt.show()

        forecast = model.predict(test_data[['ds']])
        forecast = forecast[['ds', 'yhat']]
        forecast.insert(2, "y", test_data['y'], True)
        forecast.columns = ['Dates', 'Predicted', 'Actual']
        forecast.plot(x='Dates')
        plt.ylabel("Occupancy")
        plt.ylim([0, 150])
        plt.title("3 Month Out-of-Sample Forecast")
        plt.show()

