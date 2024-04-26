import numpy as np
from scipy.signal import wiener
from scipy.fftpack import fft2, ifft2

def long_exposure_otf(u, lambda_, l, r0, alpha=0.5):
	# Calculate the long exposure OTF
	return np.exp(-1.2688 * (lambda_ * l / r0 * np.abs(u)) ** (5/3))

def wiener_deconvolution(image, kernel):
	# Perform Wiener Deconvolution
	deblurred_image = wiener(image, kernel)
	return deblurred_image

def deblur_image(image, l, r0, lambda_):
	# Calculate the spatial frequency
	u = np.fft.fftfreq(image.shape[0])

	# Calculate the long exposure OTF
	kernel = long_exposure_otf(u, lambda_, l, r0)

	# Perform Wiener Deconvolution
	deblurred_image = wiener_deconvolution(image, kernel)

	return deblurred_image

# Example usage:
image = np.array(...)  # Input image
l = 0.1  # Optic focal length
r0 = 0.01  # Fried parameter
lambda_ = 0.5  # Optical wavelength

deblurred_image = deblur_image(image, l, r0, lambda_)
