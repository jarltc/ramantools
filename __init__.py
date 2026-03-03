__version__ = "1.0.0"
__author__ = "Jarl"

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from .signal_processing import Signal

# please forgive me if I misuse the terms "signal" and "spectrum" 
# I honestly cannot tell the difference

def from_txt(file:Path, prominence:float=0.01, Si_target=520.8, **kwargs):
    ylabel = "Raman intensity"
    xlabel = "Raman shift (cm-1)"

    spectrum = pd.read_csv(file, sep="\t", names = [xlabel, ylabel])
    spectrum.set_index(xlabel, inplace=True)
    spectrum.sort_index(inplace=True)

    # separate into intensity and raman shift
    shift = spectrum.index.to_numpy()
    intensity = spectrum[ylabel].to_numpy()

    sg = Signal("", shift, intensity, Si_target=Si_target, prominence=prominence, **kwargs)

    return sg

def extract_angle(filename):
    import re
    """ Figure out angle value from filename. 
    The function expects an angle value followed by 'Grad' """
    m = re.search(r'([0-9]+)Grad', filename)
    if m:
        return float(m.group(1))
    else:
        raise(ValueError(f"Angle value not found in {filename}"))
    