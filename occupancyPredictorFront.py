# core packages - use as shortcuts
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import scipy

# from predictor model
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.serialize import model_to_json, model_from_json
import json
import itertools
import scipy


# Creating sections to organize site
siteStart = st.beta_container()
theModel = st.beta_container()
theGraphs = st.beta_container()
forClient = st.beta_container()
modelTraining = st.beta_container()
result = st.beta_container()

#saved data file so doesn't have to rerun program every time user selects different data piece
@st.cache
def get_data(filename):
	hospital_data = pd.read_csv(filename)
	return hospital_data

with siteStart:
  st.title('Kingston Health Sciences Centre Occupancy Predictor')
  st.write('The data, our model, and predictions for the KHSC bed occupancy are all visible below:')
  with st.beta_expander("The Data"):
    st.write('The data was taken out for privacy purposes')
    st.text('Data between April 21st 2014 to March 31st 2019:')

with theModel:
  with st.beta_expander("The Model"):
    st.text('This is the python model created:')
    with st.echo():
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
            # LOAD DATA - commented out since loads data immediately when user uploads csv and saves as df
            # df = pd.read_csv(file_path)

            # PREPROCESS DATA
            data = df[['date', 'occupancy']]
            data.columns = ['ds', 'y']

            # TRAIN MODEL
            self._model = Prophet(changepoint_prior_scale=0.01, seasonality_prior_scale=5,
                                  daily_seasonality=False, uncertainty_samples=0, yearly_seasonality=10)
            self._model.add_country_holidays(country_name='CA')
            self._model.fit(data)

            # SAVE MODEL
            # with open('model.json', 'w') as fout:
                # json.dump(model_to_json(self._model), fout)

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
            # LOAD DATA - commented out since loads data immediately when user uploads csv and saves as df
            # df = pd.read_csv("data.csv")

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
            # LOAD DATA - commented out since loads data immediately when user uploads csv and saves as df
            # df = pd.read_csv("data.csv")

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


with theGraphs:
  with st.beta_expander("The Graphs"):
    st.text("To interpret the data, below are a series of graphs produces from our model:")
    from PIL import Image
    img1 = Image.open("figures/Figure_1.png")
    img2 = Image.open("figures/Figure_2.png")
    img3 = Image.open("figures/Figure_3.png")
    img4 = Image.open("figures/Figure_4.png")
    img5 = Image.open("figures/Figure_5.png")
    st.image(img1,width=750, caption="Mean Absolute Percentage Error: 10.340103873417478")
    st.image(img2,width=750, caption="Root Mean Square Error: 11.31415604556463")
    st.image(img3,width=730, caption="Visualization of the data over the four years from April 21st 2014 to March 31st 2019")
    st.subheader(' ')
    st.image(img4,width=700, caption="In the above figure, visible trends of the bed occupancy over the past few years, during holidays, days of the week, and monthly are visible")
    st.image(img5,width=750, caption="The testing data used between January 1st to March 31st")

#section for reading in the csv file
with forClient:
  st.title('Future Use For Our Client')
  st.write('The client has the ability to change the forecast dates or use a different csv file with  different data so that the model can be used in the future.')  
  uploaded_file = st.file_uploader("Then please upload the CSV file to start:", type = ["csv"])
  if uploaded_file is not None:
    with st.beta_expander("Details about upload"):
        st.write(type(uploaded_file))
        file_details = {"filename":uploaded_file.name, "filetype":uploaded_file.type, "filesize":uploaded_file.size}
        st.write(file_details)
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data:")
    st.write(df)

with modelTraining:
  if uploaded_file is not None:
    st.header('Model training')
    st.write('In this section users have the ability to see different graph distributions of different features from the csv file. Users also have the ability to select different dates to run the model on.')
    st.write('')
    admissions_col, choice_col = st.beta_columns(2)
    admissions_col.write(pd.DataFrame({'Features': df.columns,}))
    input_feature = choice_col.text_input('Which feature would you like to ','occupancy')

    st.subheader('A graph of the ' + input_feature + ' over time  ')
    dist = pd.DataFrame(df[input_feature])
    st.line_chart(dist)
    st.subheader('Density of the ' + input_feature + ' feature over time ')
    st.text('Hover over the chart to determine what ' + input_feature + ' levels are most frequent')
    x1 = dist[input_feature]
    hist_data = [x1]
    group_labels = ['Actual Data']
    fig = ff.create_distplot(hist_data, group_labels)
    st.plotly_chart(fig, use_container_width=True)

with result:
  if uploaded_file is not None:
    st.header('The Predicitons:')
    startDate = st.text_input('What date would you like to start your prediction at:','2021-04-01')
    endDate = st.text_input('What date would you like to end the prediction:','2021-06-01')


if uploaded_file is not None:
  if __name__ == '__main__':
      # EXAMPLE CLASS USE
      # Initialize class
      # main()
      occupancyForecaster = OccupancyForecaster()
      # Train model with the user's data
      occupancyForecaster.train_model(str(uploaded_file.name))
      # Make predictions for a given range of dates supplied by the user
      predictions = occupancyForecaster.forecast_occupancy(startDate, endDate)


with result:
  if uploaded_file is not None:
    st.header('')
    table_col, explaination_col = st.beta_columns(2)
    input_Dates = pd.date_range(startDate, endDate) 
    table_col.write(pd.DataFrame({'Date': input_Dates,'Predicted Occupancy': predictions,}))
    explaination_col.write('To the left are the predicted hospital bed occupancy levels and their corresponding date based off of the users input.')
    st.subheader('Forecast of hospital bed occupancy between the dates ' + startDate + ' and ' + endDate)
    predictionChart = pd.DataFrame({'date': input_Dates, 'Occupancy': predictions})
    st.line_chart(predictionChart.rename(columns={'date':'index'}).set_index('index'))
