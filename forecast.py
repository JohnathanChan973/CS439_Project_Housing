import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

class Forecast:
    def __init__(self, df : pd.DataFrame, percentage : float=0.8):
        """
        Initializes the forecasting class with a dataframe and train/test split percentage
        
        Parameters:
            df (pd.DataFrame): A dataframe with time series data to forecast
            percentage (float, default=0.8): Percentage of data to use for training
        """
        self.df = df
        self.train_size = int(len(df) * percentage)
        self.train, self.test = df.iloc[:self.train_size], df.iloc[self.train_size:] # The first 80% of data is for training (by default), the remaining is for testing since past predicts future
        self.forecasts = {}
        self.metrics = {}
    
    def fit_arima(self, order : tuple=(5, 1, 0), model_name : str="ARIMA"):
        """
        Fits an ARIMA model to the training data
        
        Parameters
            order (tuple): ARIMA order parameters (p, d, q)
            model_name (str): Name to identify this model in results
        """
        model = ARIMA(self.train, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(self.test))
        
        # Stores the forecast results
        self.forecasts[model_name] = forecast
        self._calculate_metrics(model_name)
        
        return forecast
    
    def fit_holtwinters(self, trend : str="add", seasonal : str="add", seasonal_periods : int=12, model_name : str="Holt-Winters"):
        """
        Fits a Holt-Winters model to the training data
        
        Parameters:
            trend (str, default="add"): Type of trend component ('add', 'mul', or None)
            seasonal (str, default="add"): Type of seasonal component ('add', 'mul', or None)
            seasonal_periods (int): Number of time steps in a seasonal period
            model_name (str): Name to identify this model in results
        """
        model = ExponentialSmoothing(self.train, trend=trend, seasonal=seasonal, 
                                     seasonal_periods=seasonal_periods)
        model_fit = model.fit()
        forecast = model_fit.forecast(len(self.test))
        
        # Stores the forecast results
        self.forecasts[model_name] = forecast
        self._calculate_metrics(model_name)
        
        return forecast
    
    def fit_sarima(self, order : tuple=(1,1,1), seasonal_order : tuple=(1,1,1,12), model_name : str="SARIMA"):
        """
        Fits a SARIMA model to the training data
        
        Parameters:
            order (tuple): SARIMA order parameters (p, d, q)
            seasonal_order (tuple): SARIMA seasonal_order parameters (P, D, Q, S)
            model_name (str): Name to identify this model in results
        """
        model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order)
        
        model_fit = model.fit()
        forecast = model_fit.forecast(len(self.test))

        # Store forecast results
        self.forecasts[model_name] = forecast
        self._calculate_metrics(model_name)
        
        return forecast
    
    def _calculate_metrics(self, model_name : str):
        """
        Calculate error metrics for a given model's forecast
        
        Parameters:
            model_name (str): The name of the model as it is stored in self.forecasts 
        """
        forecast = self.forecasts[model_name]
        
        mae = mean_absolute_error(self.test, forecast)
        mse = mean_squared_error(self.test, forecast)
        rmse = np.sqrt(mse)
        
        self.metrics[model_name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        }
    
    def plot_forecast(self, model_name : str, figsize : tuple=(12, 6), linestyle : str="--"):
        """
        Plot a specific forecast against actual data
        
        Parameters:
            model_name (str): Name of the forecast to plot as stored in self.forecasts
            figsize (tuple): Figure size for the plot
            linestyle (str): Line style for the forecast line
        """
        if model_name not in self.forecasts:
            raise ValueError(f"Model {model_name} has not been fitted yet")
            
        forecast = self.forecasts[model_name]
        
        plt.figure(figsize=figsize)
        plt.plot(self.train, label="Training Data")
        plt.plot(self.test, label="Actual Test Data")
        plt.plot(self.test.index, forecast, label=f"{model_name} Forecast", linestyle=linestyle)
        plt.legend()
        plt.title(f"{model_name} Model Forecasting")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_all_forecasts(self, figsize : tuple=(12, 6)):
        """
        Plots all created forecasts at once for comparison
        
        Parameters:
            figsize (tuple): Figure size of the plot
        """
        plt.figure(figsize=figsize)
        plt.plot(self.train, label="Training Data", color="blue")
        plt.plot(self.test, label="Actual Test Data", color="green")
        
        colors = ["red", "purple", "orange", "brown", "pink"]
        linestyles = ["--", "-.", ":", "-"]
        
        for i, (name, forecast) in enumerate(self.forecasts.items()):
            plt.plot(self.test.index, forecast, 
                     label=f"{name} Forecast", 
                     linestyle=linestyles[i % len(linestyles)],
                     color=colors[i % len(colors)])
        
        plt.legend()
        plt.title("Forecasting Models Comparison")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def print_metrics(self, model_name : str=None):
        """
        Print metrics for a specific model or all models
        
        Parameters:
            model_name (str, default=None): Name of the model to print metrics for (if None, print all)
        """
        if model_name is not None:
            if model_name not in self.metrics:
                raise ValueError(f"Model {model_name} has not been fitted yet")
            
            metrics = self.metrics[model_name]
            print(f"=== {model_name} Metrics ===")
            print(f"{model_name} MAE: {metrics['MAE']:.4f}")
            print(f"{model_name} MSE: {metrics['MSE']:.4f}")
            print(f"{model_name} RMSE: {metrics['RMSE']:.4f}")
            print()
            
        else:
            for name, metrics in self.metrics.items():
                print(f"=== {name} Metrics ===")
                print(f"{name} MAE: {metrics['MAE']:.4f}")
                print(f"{name} MSE: {metrics['MSE']:.4f}")
                print(f"{name} RMSE: {metrics['RMSE']:.4f}")
                print()
    
    def get_best_model(self, metric : str="MSE"):
        """
        Get the name of the best performing model based on a specific metric
        
        Parameters:
            metric (str): Metric to use for comparison ('MAE', 'MSE', or 'RMSE')
            
        Returns:
            Name of the best model (str)
        
        Raises:
            ValueError: If no models have been forecasted/fitted
        """
        if not self.metrics:
            raise ValueError("No models have been fitted yet")
            
        best_model = min(self.metrics.keys(), 
                         key=lambda model: self.metrics[model][metric])
        
        return best_model