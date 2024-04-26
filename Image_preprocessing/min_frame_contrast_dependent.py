import numpy as np

def calculate_kmin(tmax, sigma2o, sigma2w, sigma2e_K, M):
	# Calculate the minimum number of frames to average
	Kmin = tmax / (sigma2e_K * (1/M * np.sum(sigma2o) + sigma2w * tmax**2))
	return Kmin

# Example usage:
tmax = 0.8  # Maximum transmission value
sigma2o = np.array([0.1, 0.2, 0.3])  # Noise variance for each pixel
sigma2w = 0.01  # Noise variance
sigma2e_K = 2e-5  # Desired noise variance
M = 1024  # Number of pixels per frame

Kmin = calculate_kmin(tmax, sigma2o, sigma2w, sigma2e_K, M)
print("Minimum number of frames to average:", Kmin)
