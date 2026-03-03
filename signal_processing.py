from . import xr
from . import np
from . import plt

from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from typing import Dict

class Signal:
    """ A single Raman spectrum. """
    def __init__(self, name:str, 
                 x:np.ndarray, 
                 y:np.ndarray, 
                 Si_target=None, 
                 prominence=0.01, 
                 peak_fn="gauss",
                 fit=False):
        self.name = name
        if len(x) != len(y):
            raise ValueError("Mismatched x and y sizes")
        
        self.x = x.copy()
        self.y = y.copy()
        self._data = self._to_da(x, y)

        # preprocess
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

    def get_peak_centers(self, prominence):
        self.prominence = prominence
        indexes, _ = find_peaks(self.y, prominence=prominence)
        centers = np.array([self.x[index] for index in indexes])
        self.peak_centers = centers
        self.peak_indexes = indexes
        
        # it might be better to use the fitted gaussians in fit_peaks to get the intensities
        # but I will keep this for now
        self.peak_intensities = [self.y[index] for index in indexes]
        
        return centers
    
    def fit_peaks(self, epsilon = {"E": (3, 8), "A": (5, 5), "Si": (6, 6)}):
        # epsilon: distances (cm-1) from peak center, must be carefully chosen for E2g peak to only fit 
        # to the rightmost portion of the peak.
        data = self._data
        fitted_peaks: Dict[str, Peak|None] = {"E":None, "A":None, "Si":None}  # type:ignore
        for i, peak_name in enumerate(["E", "A", "Si"], start=1):  # exclude LA(M) peak
            center = self.peak_centers[i]
            eps_L, eps_R = epsilon[peak_name]
            peak = Peak(data, bounds=(center-eps_L, center+eps_R), center=center, peak_fn=self.peak_fn)
            fitted_peaks[peak_name] = peak

        # Peaks have Gaussian parameters (amplitude, sigma, mu) recorded
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
        # TODO this doesnt use self.shift at all...
        # get_peak_centers must be run first before using this!
        delta = target - self.peak_centers[3]  # Si peak should be fourth from left to right
        self.x += delta
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
    
    def fit_region(self, extent:tuple[float, float], **kwargs):
        left, right = extent
        center = left + (right-left)/2
        peak = Peak(self._data, bounds=(left, right), center=center, peak_fn=self.peak_fn, **kwargs)

        return peak
    

class Peak:
    """ Gaussian peak fitting.

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
                 peak_fn="gauss",
                 baseline=0):
        self.data = (data-baseline).copy()
        self.left, self.right = bounds
        self.center = center

        peak_fn_options = ["gauss", "lorentz"]
        if peak_fn not in peak_fn_options:
            raise ValueError(f"{peak_fn} not recognized, choose from: {peak_fn_options}")
        else:
            self.peak_fn = peak_fn

        self.y = self.data.sel(x=slice(self.left, self.right)).to_numpy()
        self.x = self.data.sel(x=slice(self.left, self.right)).coords["x"].to_numpy()

        parameterstyle = {"gauss": {"amplitude":0., "sigma":0., "center":0.}, 
                          "lorentz": {"width": 0., "center":0., "intercept":0.}}
        self.parameters = parameterstyle[peak_fn]

        # attempt to fit peak
        # self.amplitude, self.sigma, self.center = self.fit_gauss()
        
        # fit and unpack results into dict
        parameters = self.fit()
        for i, key in enumerate(self.parameters):
            self.parameters[key] = parameters[i]

    @staticmethod
    def gauss(x:np.ndarray, amplitude:float, std:float, center:float):
        """ Standard Gaussian function.

        A * exp(-1/(2*sigma^2) * (x-mu)^2)

        Args:
            x (np.ndarray): x data.
            amplitude (float): Value for the amplitude.
            std (float): Value for the standard deviation.
            center (float): Value for the center of the distribution.

        Returns:
            np.ndarray: Values of the specified Gaussian given f(x)
        """
        y = amplitude*np.exp((-1/(2*std)**2)*(x-center)**2)
        return y
    
    @staticmethod
    def lorentz(x:np.ndarray, width:float, center:float, intercept:float):
        """ Lorentzian line function.
        
        See more at: https://mathworld.wolfram.com/LorentzianFunction.html

        Args:
            x (np.ndarray): x data.
            width (float): Width of the peak.
            center (float): Center for the maximum.

        Returns:
            _type_: _description_
        """
        y = intercept + 1/np.pi * 0.5*width/((x-center)**2 + (0.5*width)**2)
        return y
        
    @property
    def width(self) -> float:
        """ Calculate the full width at half maximum for the peak. 
        
        A Gaussian distribution has the full width at half maximum:
        W = 2 * sqrt(log(2)) * sigma
        """
        # C = 2*np.sqrt(2*np.log(2))
        # return C*self.sigma
        raise NotImplementedError()

    def fit(self, **kwargs):
        functions = {
            "gauss": self.fit_gauss, 
            "lorentz": self.fit_lorentz
            }
        return functions[self.peak_fn](**kwargs)

    def fit_gauss(self, method='trf'):
        """ Try to fit a Gaussian function to the data within bounds.

        Args:
            method (str, optional): Optimization method for curve_fit. Defaults to 'trf'.

        Returns:
            list[float]: Returns optimized parameters that fit the data.
        """

        center = self.center
        y0 = self.data.sel(x=center, method='nearest').item()
        result = curve_fit(self.gauss, self.x, self.y, p0=(y0, 0.5, center), method=method)
        parameters = result[0]
        return parameters

    def fit_lorentz(self, method='trf'):
        """ Try to fit a Lorentzian function to the data within bounds.

        Args:
            method (str, optional): Optimization method for curve_fit. Defaults to 'trf'.

        Returns:
            list[float]: Returns optimized parameters that fit the data.
        """

        center = self.center
        width = self.right - self.left  # take full region as initial guess
        intercept = 0
        result = curve_fit(self.lorentz, self.x, self.y, p0=(width, center, intercept), method=method, bounds=(0, np.inf))
        parameters = result[0]
        return parameters

    def get_fitted(self, x=None):
        """ Return y values evaluated at x. If x is not specified, use the given bounds. """
        if x is None:
            x = self.x
        
        if self.peak_fn == "gauss":
            amplitude = self.parameters["amplitude"]
            sigma = self.parameters["sigma"]
            center = self.parameters["center"]
            return self.gauss(x, amplitude, sigma, center)
        elif self.peak_fn == "lorentz":
            width = self.parameters["width"]
            center = self.parameters["center"]
            intercept = self.parameters["intercept"]
            return self.lorentz(x, width, center, intercept)

    # def _get_parameters(self):
    #     return self.parameters
    