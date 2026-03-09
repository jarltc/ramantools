# ramantools

Short project description: a concise summary of the repository’s purpose and scope.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Features
- Combine multiple .txt files into netCDF/DataArrays
- Peak fitting and analysis for MoS2 Raman data using Signal
- Calculate MoS2 signal metrics:
  - LA(M)/A1g ratio
  - E2g - A1g peak widths
  - E2g - A1g peak distance
- Automatically create mappings of aforementioned metrics into netCDF/DataArrays

## Installation
Install with pip (editable install)

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/<user>/ramantools.git
cd ramantools
pip install -e .
```

This installs the package into your current Python environment while allowing you to modify the source code.

## Usage

### Loading a Spectrum

```python
from ramantools import from_txt
from definitions import root

signal = from_txt(
    root / "raman" / "sample.txt",
    prominence=0.03,   # minimum peak prominence after normalization
    peak_fn="lorentz"  # "gauss" or "lorentz"
)
```

On load, the spectrum is normalized and peak centers are detected automatically.

---

### Fitting Peaks

**Standard peaks (E2g, A1g, Si):**

```python
signal.fit_peaks()

signal.fitted_peaks["E"].params   # {"height", "center", "gamma"}
signal.fitted_peaks["E"].fwhm     # full width at half maximum (cm⁻¹)

signal.EA_distance()  # distance between E2g and A1g centers
signal.LAM_ratio()    # LA(M) / A1g intensity ratio
```

**Arbitrary spectral region:**

```python
extent = (498, 541)
baseline = signal._data.sel(x=498, method='nearest').item()
peak = signal.fit_region(extent, baseline=baseline, bounds=(0, None))

peak.params  # fitted parameters
peak.fwhm    # FWHM (cm⁻¹)
```

---

### Evaluating and Plotting a Fit

```python
import numpy as np
import holoviews as hv

x = np.linspace(*extent, 500)
signal._data.hvplot(label="signal") * hv.Curve((x, peak.evaluate(x)), label="fit")
```

---

### Baseline Correction

```python
corrected = signal.correct_baseline(niter=20)  # returns a new Signal
corrected.peak_fn = "lorentz"
peak = corrected.fit_region((210, 230))
```

> Inspect results visually — correction can be unreliable near overlapping peaks.
