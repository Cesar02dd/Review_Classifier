import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import norm


class SupervisedLearning:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data_train = self.data_loader.data_train.select_dtypes(include=['number'])
        self._labels_train = self.data_loader.labels_train
        self._data_test = self.data_loader.data_test.select_dtypes(include=['number'])
        self._labels_test = self.data_loader.labels_test

    def bayesian_linear_regression(self):
        prior_mean = np.zeros(self._data_train.shape[1])
        prior_covariance = np.eye(self._data_train.shape[1])

        # Calculate prior distribution
        prior_distribution = norm.pdf(np.linspace(-3, 3, 100), loc=prior_mean[1], scale=np.sqrt(prior_covariance[1, 1]))

        # Estimate the likelihhod variance from the data using the empirical variance of the residuals (differences between the targets and predicted values).
        model = LinearRegression()
        model.fit(self._data_train, self._labels_train)  # Fit the model
        y_train_pred = model.predict(self._data_train)  # Predict on the training set
        residuals = self._labels_train - y_train_pred  # Calculate residuals
        likelihood_variance = np.var(residuals)  # Estimate likelihood variance

        # Gaussian likelihood
        likelihood_distribution = norm.pdf(self._labels_train, loc=0, scale=np.sqrt(likelihood_variance))

        # Calculate posterior parameters
        posterior_covariance = np.linalg.inv(
            np.linalg.inv(prior_covariance) + (1 / likelihood_variance) * self._data_train.T @ self._data_train)
        posterior_mean = (1 / likelihood_variance) * posterior_covariance @ self._data_train.T @ self._labels_train

        # Calculate posterior distribution
        posterior_distribution = norm.pdf(np.linspace(-3, 3, 100), loc=posterior_mean[1],
                                          scale=np.sqrt(posterior_covariance[1, 1]))

        # Draw samples from the posterior distribution
        num_samples = 1000
        posterior_samples = np.random.multivariate_normal(posterior_mean, posterior_covariance, size=num_samples)

        # Calculate predictions using samples
        y_pred_samples = self._data_test @ posterior_samples.T

        # Calculate mean and standard deviation of predictions
        y_pred_mean = np.mean(y_pred_samples, axis=1)
        print("Mean of Prediction Sample: ", y_pred_mean)
        y_pred_std = np.std(y_pred_samples, axis=1)
        print("Standard Deviation of Prediction Sample: ", y_pred_std)

        # Assess performance (MSE and R2) and print it
        mse = mean_squared_error(self._labels_test, y_pred_mean)
        r2 = r2_score(self._labels_test, y_pred_mean)

        # Plot the results
        print("Mean Squared Error: ", mse)
        print("R-squared: ", mse)

        # Plot prior distribution
        plt.figure(figsize=(15, 10))
        pl.subplot(3, 1, 1)
        plt.plot(np.linspace(-3, 3, 100), prior_distribution, label="Prior")
        plt.legend()

        # Plot likelihood distribution
        pl.subplot(3, 1, 2)
        plt.scatter(self._labels_train, likelihood_distribution, label="likelihood")
        plt.legend()

        # Plot posterior distribution
        pl.subplot(3, 1, 3)
        plt.plot(np.linspace(-3, 3, 100), posterior_distribution, label="posterior")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot y_test vs y_pred with error bars
        plt.figure()
        plt.errorbar(self._labels_test, y_pred_mean, yerr=2 * y_pred_std, linestyle='', color='green', alpha=0.6)
        plt.show()
