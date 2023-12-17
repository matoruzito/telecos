# INDICADOR DE VERSION MANUAL: V1.1
# LAST EDITOR: MATIAS
import numpy as np


# This method returns two vectors.
# The first one is the vector of frequencies and the second one is the spectrum of the input signal.
def fourier(x_, samplingFreq_):
    x_ = np.fft.fftshift(np.fft.fft(x_))
    # Definimos vector de frecuencias
    sampNum = x_.size
    if np.remainder(sampNum, 2) == 0:
        k = np.arange(-sampNum/2, sampNum/2, 1)
    else:
        k = np.arange(-(sampNum-1)/2-1, (sampNum-1)/2, 1)
    frequencies = k * samplingFreq_ / sampNum
    spectrum = x_ / samplingFreq_
    return frequencies, spectrum


def antifourier(x_, samplingFreq_):
    return np.fft.ifft(np.fft.ifftshift(samplingFreq_ * x_))


def signal_cajon(x_, a=-0.5, b=0.5):
    return np.where((x_ >= a) & (x_ <= b), 1, 0)


def signal_triangulo(x_):
    return (1 - np.abs(x_))*(np.abs(x_) < 1)


def signal_mensaje(t_, W_=1, M_=1):
    m = 2.0 * W_ * np.sinc(2.0 * W_ * t_) - W_ * np.sinc(W_ * t_)**2
    m = M_ * m / np.max(m)
    return m


def signal_coseno(t_, fc_=10, A_=1):
    return A_ * np.cos(2*np.pi*fc_*t_)


def sys_hilbert(x_):
    return -1j*np.sign(x_)


# Integrador discreto. Recibe el eje de tiempos y la señal discreta x
def sys_integrador(t_, x_):
    samplingTime = t_[1] - t_[0]
    integral = np.zeros(np.size(t_))
    integral[0] = x_[0] * samplingTime
    for i in range(1, np.size(t_)):
        integral[i] = integral[i-1] + x_[i] * samplingTime
    return integral


# Calcula la DEP de la secuencia x_ (Una realizacion)
def get_dep(x_, samplingTime_):
    numMuestras = len(x_)
    F, X = fourier(x_, 1 / samplingTime_)
    dep = (1 / samplingTime_) ** 2 * np.abs(X) ** 2
    dep = samplingTime_ * dep / numMuestras
    return F, dep


# Función que devuelve la estimación de la DEP de un proceso
def get_depEstimation(processName_, *args_, numSamples_, samplingTime_, numRealizations_):
    depEstimada = np.zeros(numSamples_)
    for i in range(numRealizations_):
        realizationI = processName_(*args_)
        F, DEPI = get_dep(realizationI, samplingTime_)
        depEstimada = depEstimada + DEPI
    return F, depEstimada / numRealizations_

# Función que genera una realización de ruido blanco


def get_whiteNoiseRealization(numSamples_, n0_, samplingTime_):
    return np.sqrt(n0_ / samplingTime_) * np.random.randn(numSamples_)
