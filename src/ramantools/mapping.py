from . import xr
from . import np
from . import plt
from . import Path

from typing import Dict, Any
import plotly.graph_objects as go
import pandas as pd
import re

from .signal_processing import Signal

class MapArray:
    # TODO separate txt extraction function
    def __init__(self, folder:Path, name:str) -> None:
        self.path_to_data = folder
        self.name = name
        self.ylabel = "Raman intensity"
        self.xlabel = "Raman shift (cm-1)"
        pass

    def read_dataframe(self, file:Path) -> pd.DataFrame:
        """ Load .txt files into a DataFrame.

        Args:
            file (Path): Path to a single .txt file.

        Returns:
            pd.DataFrame: DataFrame containing Raman data.
        """
        df = pd.read_csv(file, sep="\t",
                         names = ["x", self.xlabel, self.ylabel])
        return df

    def create_mapping_dataarray(self):
        """ Main calculation loop for turning .txt files into DataArrays, and combines them at the end.
        """
        folder = self.path_to_data
        dataarrays = []

        # loop through txt files, containing data for a single angle
        for file in folder.glob("*.txt"):
            angle = self.extract_angle(file.stem)
            df = self.read_dataframe(file)
            dataarrays.append(self.process_df(df, angle))  # convert table to DataArray

        full_da = xr.concat(dataarrays, dim="angle")
        self.da = full_da.sortby(["angle", "position"])

    def process_df(self, df:pd.DataFrame, angle:float):
        """Create DataArrays from a single angle scan.

        A single DataFrame (from the raw .txt file) contains spectra for several positions.
        This splits the Raman data into spectra for each position.

        Args:
            df (pd.DataFrame): DataFrame containing spectra along a single angle.
            angle (float): Angle where the measurement was taken.
        """
        # split dataframe into positions x
        dataarrays = []
        df["x"] /= 1000
        gb = df.groupby("x")  # group data by position

        # loop through positions
        for position in gb.groups:
            spectrum = gb.get_group(position).set_index(self.xlabel)
            spectrum.sort_index(inplace=True)
            
            # separate into intensity and raman shift
            shift = spectrum.index.to_numpy()
            intensity = spectrum[self.ylabel].to_numpy()

            # create DataArray for a single angle
            angle_da = xr.DataArray(intensity,
                              coords={"angle": angle, "position": position, "raman_shift":shift},
                              dims=["raman_shift"],
                              name="intensity")
            dataarrays.append(angle_da)
        
        # compile multiple DataArrays into a single one
        angle_da = xr.concat(dataarrays, dim="position")

        return angle_da

    @staticmethod
    def extract_angle(filename):
        """ Figure out angle value from filename. 
        The function expects an angle value followed by 'Grad' """
        m = re.search(r'([0-9]+)Grad', filename)
        if m:
            return float(m.group(1))
        else:
            raise(ValueError(f"Angle value not found in {filename}"))
        

class Calculation:
    def __init__(self, da:xr.DataArray, 
                 Si_target=520.0, 
                 peak_prominence=0.02,
                 wafer="N/A") -> None:
        self.da = da
        self.Si_target = Si_target
        self.peak_prominence = peak_prominence

        self.angles = self.da.coords["angle"].values
        self.positions = self.da.coords["position"].values

        self.dataarrays: Dict[str, Any] = {
            "E_width": None,
            "A_width": None,
            "EA_distance": None,
            "LAM_ratio": None,
        }


    def process_signal(self, da:xr.DataArray, angle:float, position:float):
        """ Extract Raman spectrum from a single position, and use it to create Signals for processing.

        Args:
            da (xr.DataArray): DataArray containing spectra at various positions and angles.
            angle (float): Angle where the spectrum should be taken.
            position (float): Position where the spectrum should be taken.

        Returns:
            Signal: Raman signal shifted to the Si target and with Gaussians fit to the E, A, and Si peaks.
        """
        # create dummy Signal for processing
        spectrum = da.sel(angle=angle, position=position).to_numpy()
        signal = Signal("", 
                        da.coords["raman_shift"].values, 
                        spectrum,
                        Si_target=self.Si_target)
        signal.get_peak_centers(self.peak_prominence) 
        signal.fit_peaks() 
        return signal


    def calculate_parameters(self):
        """ Loop through angles and positions in the dataset, fit peaks, and calculate relevant data.
        """
        da = self.da

        # angle data contains arrays of position_data
        angle_data = []
        for angle in self.angles:
            # sweep through positions and store values in an array
            position_data = []
            for position in self.positions:
                # each position should have 4 calculated values
                signal = self.process_signal(da, angle, position)
                widths = signal.fwhm()
                Ewidth, Awidth = widths["E"], widths["A"]
                distance = signal.EA_distance() if (signal.EA_distance() < 30.) else np.nan
                ratio = signal.LAM_ratio()

                data = np.array([Ewidth, Awidth, distance, ratio])
                position_data.append(data)

            positions_np = np.stack(position_data)  # convert to np array
            angle_data.append(positions_np)

        # create a new axis and combine arrays into one array of shape (n_angles, n_positions)
        map_data = np.stack(angle_data)
        assert map_data.shape == (8, 5, 4)  # 8 angles, 5 positions, 4 variables 
        
        # initialize dataarray
        names = ["E_width", "A_width", "EA_distance", "LAM_ratio"]
        for index, variable in enumerate(names):
            # convert data for each variable into DataArrays
            da = self.build_dataarray(names[index], map_data[:, :, index])
            self.dataarrays[variable] = da

    
    def compile_dataset(self) -> xr.Dataset:
        """ Combine a dict of DataArrays (self.dataarrays) into a Dataset. """
        ds = xr.Dataset(self.dataarrays)
        return ds
    

    def build_dataarray(self, variable:str, arr:np.ndarray):
        # arr should have shape 8, 5
        angles = self.angles
        positions = self.positions

        assert arr.shape == (8, 5)
        assert len(angles) == 8
        assert len(positions) == 5

        da = xr.DataArray(arr, 
                          dims=("angle", "position"), 
                          coords={"angle": angles,
                                  "position": positions}, 
                          name=variable)
        
        return da


class RamanMappingPlot:
    # TODO modify to use DataArray methods (i dont know if this is different)
    # TODO move to plotting
    def __init__(self, da:xr.DataArray, title:str):
        """ Create plot for Raman mapping.

        Args:
            da (xr.DataArray): DataArray containing a map for a single variable.
            title (str): Title for the plot.
        """
        self.da = da.copy()
        self.title = title
        self.azimuths = np.deg2rad(self.da.coords["angle"].values)
        self.zeniths = da.coords["position"].values

        # collapse value at the center
        self.center = self.da.sel(position=0).mean(dim="angle").item()
        self.da.loc[dict(position=0)] = self.center

        self.titles = {"E_width": "$E_{2g}$ width",
                       "A_width": "$A_{1g}$ width",
                       "LAM_ratio": "$LA(M)/A_{1g}$ ratio",
                       "EA_distance": "$E_{2g}$ $A_{1g}$ distance"}
        
        self.units = {"E_width": "$\\text{cm}^{-1}$",
                      "A_width": "$\\text{cm}^{-1}$",
                      "LAM_ratio": "a.u.",
                      "EA_distance": "$\\text{cm}^{-1}$"}

    def plot_mpl(self, contour=False):
        # TODO set figsize
        da = self.da
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        ax.set_title(self.titles[self.title])

        # create meshgrid
        r, theta = np.meshgrid(self.zeniths, self.azimuths)
        values = da.to_numpy()

        if contour:
            sc = ax.contourf(theta, r, c=values, zorder=3)
        else:
            sc = ax.scatter(theta, r, c=values, s=100, zorder=3)
        plt.colorbar(sc, label=self.units[self.title])

        ax.set_rlim(0, 100) # type: ignore
        
        return fig

    def plot_plotly(self, outfile="polar_plot.html"):
        raise NotImplementedError
        # da = self.da

        # r, theta = np.meshgrid(self.zeniths, self.azimuths)
        # values = da["E_width"].to_numpy()

        # fig = go.Figure(
        #     go.Scatterpolar(
        #         r=r.flatten(),
        #         theta=np.rad2deg(theta.flatten()),  # Plotly expects degrees
        #         mode="markers",
        #         marker=dict(
        #             color=values.flatten(),
        #             colorscale="Viridis",
        #             size=8,
        #             colorbar=dict(title="E_width"),
        #         ),
        #         customdata=values.flatten(),
        #         hovertemplate=(
        #             "r: %{r}<br>"
        #             "theta: %{theta}<br>"
        #             "E_width: %{customdata}<extra></extra>")
        #     )
        # )

        # fig.update_layout(
        #     polar=dict(radialaxis=dict(range=[0, 100])),
        #     showlegend=False,
        # )

        # fig.write_html(outfile)
        # return fig

###### functions ######
# TODO move to __init__ base ramantools
def signal2map(da: xr.DataArray, outdir:Path|None=None, fname="mapping_gauss", **kwargs):
    calculation = Calculation(da, **kwargs)
    calculation.calculate_parameters()
    ds = calculation.compile_dataset()

    if outdir is not None:
        ds.to_netcdf(outdir/f"{fname}.nc")

def compile_da(wafer_name, raw_data, outdir):
    """Compile raw Raman spectra (.txt) from a mapping into a single netCDF file.

    Assumes that the data for multiple angles is stored in a single .txt file.

    Args:
        wafer_name (str): Wafer name
        raw_data (Path): Path to a single folder containing the raw data. 
            The angle value is inferred from the file name ("xxGrad").
        outdir (Path): File will be saved in outdir/wafer_name.nc
    """
    if not outdir.exists():
        outdir.mkdir(parents=True)

    dataset = MapArray(raw_data, name=wafer_name)
    dataset.create_mapping_dataarray()
    dataset.da.to_netcdf(outdir/f"{wafer_name}.nc")
    print(f"Raman mapping for wafer converted to DataArray {outdir}/{wafer_name}.nc")
