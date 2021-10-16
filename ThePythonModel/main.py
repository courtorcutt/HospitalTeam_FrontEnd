from occupancyforecaster import OccupancyForecaster

if __name__ == '__main__':
    # EXAMPLE CLASS USE
    # Initialize class
    occupancyForecaster = OccupancyForecaster()
    # Train model with the user's data
    occupancyForecaster.train_model("data.csv")
    # Make predictions for a given range of dates supplied by the user
    predictions = occupancyForecaster.forecast_occupancy("2021-04-01", "2021-06-01")
    # Display predictions to the user
    print("Predictions:", predictions)
    occupancyForecaster.test_model()
