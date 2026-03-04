from . import xr
from . import np
from . import plt

from abc import ABC, abstractmethod

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from typing import Dict
    
class Peak(ABC):
    """ Peak fitting class.

    Two functions are available: either a Gaussian peak or Lorentzian peak can be fit to the data.

    Uses xr.DataArray to simplify slicing by accessing y values using x (and not having to fiddle with indices).
    After fitting, Peaks have attributes amplitude, sigma, and center. 
    These can also be accessed with the property Peak.parameters.

    Args:
    data (xr.DataArray): One-dimensional data for intensity (y) as a function of Raman shift (x).
    bounds (tuple[left, right]): Left and right extents of the peak to be fitted.
    center: Initial guess for the center of the peak. I suggest to use scipy's find_peaks to locate this.
    """
    def __init__(self, data:xr.DataArray, 
                 bounds:tuple[float, float], 
                 center:float,
                 baseline=0):
        self.data = data.copy()
        self.left, self.right = bounds
        self.center = center
        self.baseline = baseline

        self.y = self.data.sel(x=slice(self.left, self.right)).to_numpy()
        self.x = self.data.sel(x=slice(self.left, self.right)).coords["x"].to_numpy()

        self._params = None

    def initial_guess(self):
        x = self.x
        y = self.y
        height = np.max(y)
        center = x[np.argmax(y)]
        sigma = self.right - self.left
        return (height, center, sigma)

    # abstract methods are to be handled by subclasses
    @abstractmethod
    def model(self, x, *params):
        """ Peak model function to be implemented by subclasses. """
        pass
    
    @property
    @abstractmethod
    def fwhm(self):
        """ Return full width at half maximum """
        pass

    @property
    @abstractmethod
    def params(self) -> dict:
        """ Return fitting parameters as dict """
        pass

    def fit(self, **kwargs):
        """ Fit curve to the data.

        kwargs are passed to curve_fit().

        Returns:
            tuple[float, float, float]: Fitting parameters.
        """
        x = self.x
        y = self.y - self.baseline
        p0 = self.initial_guess()
        parameters, pcov = curve_fit(self.model, x, y, p0=p0, **kwargs)
        self.err = np.sqrt(np.diag(pcov))
        self._params = parameters
        return parameters

    def evaluate(self, x=None):
        """ Return y values evaluated at x. If x is not specified, use the given bounds. """
        if x is None:
            x = self.x

        if self._params is None:
            raise RuntimeError("Peak has not yet been fitted.")
        
        return self.model(x, *self._params) + self.baseline  # type:ignore
    

class GaussPeak(Peak):
    def __init__(self, data: xr.DataArray, 
                 bounds: tuple[float, float], 
                 center: float,
                 baseline=0):
        super().__init__(data, bounds, center, baseline)

    def model(self, x, height, center, sigma):
        return height * np.exp(-((x - center)**2) / (2 * sigma**2))
    
    @property
    def params(self):
        if self._params is None:
            raise RuntimeError("Peak has not yet been fitted.")
        
        height, center, sigma = self._params
        return {"height": height, "center": center, "sigma": sigma}
    
    @property
    def fwhm(self):
        sigma = self.params["sigma"]
        C = 2*np.sqrt(2*np.log(2))
        return abs(C*sigma)


class LorentzPeak(Peak):
    def __init__(self, data: xr.DataArray, 
                 bounds: tuple[float, float], 
                 center: float,
                 baseline=0):
        super().__init__(data, bounds, center, baseline)

    def model(self, x, amplitude, center, gamma):
        """ Lorentzian line function.
        
        See more at: https://mathworld.wolfram.com/LorentzianFunction.html
        """
        return amplitude * (gamma**2 / ((x-center)**2 + gamma**2))
    
    @property
    def params(self):
        if self._params is None:
            raise RuntimeError("Peak has not yet been fitted.")
        
        height, center, gamma = self._params
        return {"height": height, "center": center, "gamma": gamma}
    
    @property
    def fwhm(self):
        gamma = self.params["gamma"]
        return abs(2*gamma)

class Signal:
    """ A single Raman spectrum. """
    def __init__(self, name:str, 
                 x:np.ndarray, 
                 y:np.ndarray, 
                 Si_target=None, 
                 prominence=0.01, 
                 peak_fn="gauss",
                 fit=False,
                 preprocess=True):
        self.name = name
        if len(x) != len(y):
            raise ValueError("Mismatched x and y sizes")
        
        self.x = x.copy()
        self.y = y.copy()
        self._data = self._to_da(x, y)

        # preprocess
        if preprocess:
            self.normalize()
            self.get_peak_centers(prominence)  # allows fit_peaks to be run
            if Si_target is not None:
                self.correct_Si(Si_target)

        self.peak_fn = peak_fn

        if fit:
            self.fit_peaks()
    
    
    def _to_da(self, xdata:np.ndarray, ydata:np.ndarray):
        da = xr.DataArray(name="intensity", data=ydata, coords={"x": xdata})
        return da

    # TODO move to plotting library
    def plot(self, ax=None, color="black", limits=(100, 1100), **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.x, self.y, color=color, label=self.name, **kwargs)
        ax.grid()
        ax.set_xlim(*limits)
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_ylabel("Raman intensity (a.u.)")

        return fig

    def shift(self, value:float, inplace=False):
        if inplace:
            self.x += value
        else:
            x = self.x
            y = self.y
            result = Signal(self.name, x+value, y) 
            return result        

    def get_peak_centers(self, prominence:float):
        """ Rough peak finding algorithm.

        Uses find_peaks() from scipy.signal.

        Args:
            prominence (float): Criterion for what is considered a peak.

        Returns:
            list[float]: List of positions where the peaks are found.
        """
        self.prominence = prominence
        indexes, _ = find_peaks(self.y, prominence=prominence)
        centers = np.array([self.x[index] for index in indexes])
        self.peak_centers = centers
        self.peak_indexes = indexes
        
        # it might be better to use the fitted Peaks in fit_peaks to get the intensities
        # but I will keep this for now
        self.peak_intensities = [self.y[index] for index in indexes]
        
        return centers
    
    def fit_peaks(self, epsilon = {"E": (3, 8), "A": (5, 5), "Si": (6, 6)}):
        # epsilon: distances (cm-1) from peak center, must be carefully chosen for E2g peak to only fit 
        # to the rightmost portion of the peak.
        fitted_peaks: Dict[str, Peak|None] = {"E":None, "A":None, "Si":None}  # type:ignore
        for i, peak_name in enumerate(["E", "A", "Si"], start=1):  # exclude LA(M) peak
            center = self.peak_centers[i]
            eps_L, eps_R = epsilon[peak_name]
            bounds = (center-eps_L, center+eps_R)
            peak = self._get_peakfn(center, bounds=bounds)
            peak.fit()
            fitted_peaks[peak_name] = peak

        # Peaks have parameters (amplitude, sigma/gamma, mu) recorded
        self.fitted_peaks = fitted_peaks

    def fwhm(self) -> dict[str, float]:
        widths = {peak: np.nan for peak in ["E", "A", "Si"]}
        for peak_name in widths.keys():
            widths[peak_name] = self.fitted_peaks[peak_name].width  # type:ignore
        return widths

    def normalize(self):
        y_norm = np.sqrt((self.y**2).sum())
        self.y /= y_norm

    def correct_Si(self, target:float):
        # run once in case it wasn't yet run
        self.get_peak_centers(self.prominence)

        # look for the closest peak center to Si peak (520.8)
        eps_old = 999.
        for peak_x in self.peak_centers:
            distance = 520.8 - peak_x
            if distance < eps_old:
                Si_peak = peak_x
                eps_old = distance
            
        delta = target - Si_peak  
        self.x += delta  # shift x-values so that Si_peak matches the target position
        self.get_peak_centers(self.prominence)
        self._data = self._to_da(self.x, self.y)

    def LAM_ratio(self):
        # LA(M) peak is the first (0) and A1g is the third (2)
        ratio = self.peak_intensities[0]/self.peak_intensities[2] 
        return ratio
    
    def EA_distance(self):
        # this uses centers of the fitted gaussians (as opposed to self.peak_centers)
        Epeak = self.fitted_peaks["E"]
        Apeak = self.fitted_peaks["A"]
        delta = Apeak.center - Epeak.center  # type:ignore
        return delta
    
    def fit_region(self, extent:tuple[float, float], method='trf', baseline=0, peak_fn=None, **kwargs):
        left, right = extent
        center = left + (right-left)/2

        peak = self._get_peakfn(center, extent, baseline, peak_fn)
        peak.fit(method=method, **kwargs)

        return peak
    
    def _get_peakfn(self, center:float, bounds:tuple[float, float], baseline=0, peak_fn=None, **kwargs) -> Peak:
        """ Get peak based on the specified peak function. """
        left, right = bounds
        peak_functions = {"gauss": GaussPeak, "lorentz": LorentzPeak}

        if peak_fn is None:
            peak_fn = self.peak_fn
        
        peak = peak_functions[peak_fn](self._data, 
                                       bounds=(left, right), 
                                       center=center, 
                                       baseline=baseline, 
                                       **kwargs)
        return peak
    
    def extract_baseline(self, niter=20) -> np.ndarray:
        """ Baseline function for data correction/peak fitting.

        Taken from stackoverflow (https://stackoverflow.com/questions/57350711/baseline-correction-for-spectroscopic-data?rq=3), which
        itself was based on the following paper: https://www.caen.it/wp-content/uploads/2017/10/ED3163_gamma_spectroscopy_CAEN_edu_kit.pdf
        
        I don't quite understand how it works so proceed with caution.

        Args:
            ydata (np.ndarray): Raman y-data.
            niter (int, optional): Number of iterations. The best value is 10-20 based on my own testing.

        Returns:
            np.ndarray: Array of the calculated baseline. 
        """
        ydata = self._data.to_numpy()
        raman_spectra_transformed = np.log(np.log(np.sqrt(ydata +1)+1)+1)

        working_spectra = np.zeros(ydata.shape)

        for pp in np.arange(0, niter):
            r1 = raman_spectra_transformed
            r2 = (np.roll(raman_spectra_transformed, -pp, axis=0) + np.roll(raman_spectra_transformed, pp, axis=0))/2
            working_spectra = np.minimum(r1,r2)
            raman_spectra_transformed = working_spectra

        baseline = (np.exp(np.exp(raman_spectra_transformed)-1)-1)**2 -1
        return baseline
    
    def correct_baseline(self, niter=20):
        """ Apply baseline correction and generate new Signal instance.

        This doesn't work very well in my experience but I'm leaving it here anyway.

        Args:
            niter (int, optional): Number of iterations for baseline extraction algorithm. Defaults to 20.

        Returns:
            Signal: Signal with corrected baseline.
        """
        corrected = self._data.to_numpy().copy()
        baseline = self.extract_baseline(niter)
        corrected -= baseline
        x = self._data.coords["x"].to_numpy().copy()
        new_signal = Signal("", 
                            x, 
                            corrected, 
                            prominence=self.prominence, 
                            peak_fn=self.peak_fn)
        return new_signal
    