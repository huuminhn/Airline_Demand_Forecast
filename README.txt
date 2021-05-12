Airline Demand Forecasting
Data collected from Airline survey in 2012
Data had been cleaned

########
Purpose of this project is Prediction:
Apply booking pattern estimated from training data into validation data
Compute final demand forecasts for 7 departure dates and days 

#######
Three variable
Departure date
Booking date
Cumulative bookings

######
Training Data
84 departure dates
61 days prior (days prior=0,1,2,…,60)

Validation Data
7 departure dates (7/25~7/31)
29 days prior (days prior=0,1,…,28)
Naïve forecast

######
Booking Model Based on two information.
Booking curve : Estimated from historical bookings.
Bookings-on-hand (on-the-book): cumulative bookings at the given booking date. 

######
Additive model
Forecast = Forecasts for remaining demand + on-the-book

Multiplicative Model
Forecast = on-the-book/(historical booking rate for given days prior)

#####
Model Evaluation based on Mean Absolute Scaled Error (MASE):
(Total absolute error of model) / (Total absolute error of Naive Forecasts)

