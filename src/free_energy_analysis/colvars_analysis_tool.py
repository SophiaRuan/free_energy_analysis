import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata



plt.rcParams.update({
        'font.size': 16,              # Set default font size
        'axes.labelweight': 'bold',   # Bold labels
        'axes.linewidth': 1.5,        # Axis line width
        'grid.color': '#888888',      # Gray grid lines
        'grid.linestyle': '--',       # Dashed grid lines
        'grid.linewidth': 0.5,        # Grid line width
        'xtick.major.size': 5,        # Major tick size
        'ytick.major.size': 5         # Major tick size
    })

class ColvarsAnalyzer:
    def __init__(self,
                 base_dir,
                 number_of_cv=1,
                 cv_labels=None,
                 replica_range=range(1,11),
                 pmf_filename="colvar.out.pmf",
                 colvar_filename="colvar.out.colvars.traj"
                 ):
        """
        Initialize the ColvarsAnalyzer class.
        Identify colvars, pmf files, colvar trajectories for the following analysis.

        Parameters:
        - base_dir: Directory containing all replicas.
        - number_of_cv: Number of collective variables (CVs).
        - cv_labels: Optional list of CV names.
        - replica_range: Range or list of replica indices.
        - pmf_filename: Filename for PMF output.
        - colvar_filename: Filename for colvars trajectory.
        """
        self.base_dir = base_dir
        self.number_of_cv = number_of_cv
        # if there is no CV labels provided, then use CV_1, CV_2, CV_3 to represent CVs
        self.cv_labels = cv_labels or [f"CV_{i + 1}" for i in range(number_of_cv)]
        if len(self.cv_labels) != self.number_of_cv:
            raise ValueError("Length of cv_labels must match number_of_cv")
        self.directories = [os.path.join(base_dir, f"{i:02}_IDNR") for i in replica_range]
        self.pmf_filename = pmf_filename
        self.colvar_filename = colvar_filename
        self.pmf_files = [os.path.join(d, pmf_filename) for d in self.directories] # they are all the same pmf files
        self.colvar_files = [os.path.join(d, colvar_filename) for d in self.directories] # colvar trajectory files are different among replicas

    @staticmethod
    def read_data(path):
        """Read a data file and handle any trailing null rows."""
        data = pd.read_csv(path, comment='#', sep=r'\s+', header=None)
        if data.iloc[-1].isnull().any():
            data = data.iloc[:-1]
        return data

    def plot_pmf(self):
        """
        Plot the Potential of Mean Force (PMF) based on the number of CVs.
        Saves the resulting plot as an image.
        """
        if self.number_of_cv == 1:
            self._plot_pmf_1cv()
        elif self.number_of_cv == 2:
            self._plot_pmf_2cv()
        elif self.number_of_cv == 3:
            self._plot_pmf_3cv()
        else:
            raise ValueError("Invalid number of CVs. Only 1, 2, or 3 are supported.")

    def _plot_pmf_1cv(self):
        """Plot PMF for 1 CV."""
        data = self.read_data(self.pmf_files[0])
        plt.plot(data[0], data[1], label="Free Energy (kBT)")
        plt.xlabel(f"{self.cv_labels[0]}")
        plt.ylabel("Free Energy (kBT)")
        plt.title("Potential of Mean Force (1 CV)")
        plt.legend()
        plt.grid(True)
        plt.savefig("PMF_1CV.png", bbox_inches="tight")
        plt.close()

    def _plot_pmf_2cv(self):
        """Plot PMF for 2 CVs."""
        data = self.read_data(self.pmf_files[0])
        print(data)
        x, y, pmf_vals = data.iloc[:, 0].values, data.iloc[:, 1].values, data.iloc[:, 2].values

        grid_x = np.linspace(np.min(x), np.max(x), 100)
        grid_y = np.linspace(np.min(y), np.max(y), 100)
        X, Y = np.meshgrid(grid_x, grid_y)

        # Interpolate PMF values onto grid
        Z = griddata((x, y), pmf_vals, (X, Y), method='cubic')

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, Z, levels=30, cmap='viridis')
        cbar = plt.colorbar(contour)
        cbar.set_label("Free Energy (kBT)")

        plt.xlabel(self.cv_labels[0])
        plt.ylabel(self.cv_labels[1])
        plt.title("Potential of Mean Force (2 CVs)")
        plt.grid(True)
        plt.savefig("PMF_2CV.png", bbox_inches="tight")
        plt.close()
        # print(self.base_dir)
        # print(self.pmf_filename)
        # obj = pmf.MultipleReplica(pmf_path=self.base_dir, pmf_name=self.pmf_filename)
        # obj.plot_colvars_data(con_steps=10)
        # plt.title("Potential of Mean Force (2 CV)")
        # plt.savefig("PMF_2CV.png", bbox_inches="tight")
        # plt.close()

    def _plot_pmf_3cv(self):
        """Generate 3D PMF projections for 3 CVs and save the plot."""
        data = self.read_data(self.pmf_files[0])
        x, y, z, pmf_vals = data.iloc[:, 0].values, data.iloc[:, 1].values, data.iloc[:, 2].values, data.iloc[:, 3].values
        grid_x, grid_y, grid_z = np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100), np.linspace(min(z), max(z), 100)
        pmf_xy = griddata((x, y), pmf_vals, (grid_x[None, :], grid_y[:, None]), method='cubic')
        pmf_xz = griddata((x, z), pmf_vals, (grid_x[None, :], grid_z[:, None]), method='cubic')

        fig, axes = plt.subplots(1, 2, figsize=[20, 8])
        vmin, vmax = np.nanmin(pmf_vals), np.nanmax(pmf_vals)
        im_xy = axes[0].imshow(pmf_xy, extent=(min(x), max(x), min(y), max(y)), origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(im_xy, ax=axes[0]).set_label(r'$\Delta G$ [kT]', fontsize=20)
        axes[0].set_xlabel(self.cv_labels[0])
        axes[0].set_ylabel(self.cv_labels[1])
        axes[0].set_title('XY Projection')

        im_xz = axes[1].imshow(pmf_xz, extent=(min(x), max(x), min(z), max(z)), origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(im_xz, ax=axes[1]).set_label(r'$\Delta G$ [kT]', fontsize=20)
        axes[1].set_xlabel(self.cv_labels[0])
        axes[1].set_ylabel(self.cv_labels[2])
        axes[1].set_title('XZ Projection')

        plt.tight_layout()
        plt.savefig("PMF_3CV.png", bbox_inches="tight")
        plt.close()

    def plot_cv_histogram_full(self):
        """
        Plot histograms of all CVs in a single figure with stacked subplots.
        """
        # Read data from all colvar trajectories and concatenate all data into one DataFrame
        all_data = [self.read_data(f) for f in self.colvar_files if os.path.isfile(f)]
        all_data = pd.concat(all_data)

        figs, axes = plt.subplots(self.number_of_cv, 1, figsize=(10, self.number_of_cv * 5))  # Create subplots
        if self.number_of_cv == 1:
            axes = [axes]

        for i, label in enumerate(self.cv_labels):
            axes[i].hist(all_data.iloc[:, i + 1], bins=20, alpha=0.7, label=label)
            axes[i].set_xlabel("CV Value")
            axes[i].set_ylabel("Frequency")
            axes[i].legend()
            axes[i].grid(True)

        figs.suptitle("CV Histograms", fontsize=20)
        plt.tight_layout()
        figs.savefig("CV_Histograms.png", bbox_inches="tight")
        plt.close()

    def plot_traj_overtime_all(self):
        colvar_data_all = [self.read_data(f) for f in self.colvar_files if os.path.isfile(f)]
        n = self.number_of_cv
        figs, axes = plt.subplots(n + 1, 1, figsize=(15, 15))
        axes = axes.flatten()

        for i in range(n):
            for j, data in enumerate(colvar_data_all):
                axes[i].plot(data.iloc[:, 0], data.iloc[:, i + 1], label=f'{self.cv_labels[i]} Replica {j + 1}')
            axes[i].set_xlabel("Time Step (fs)")
            axes[i].set_ylabel(self.cv_labels[i])
            axes[i].legend()
            axes[i].grid(True)

        for j, data in enumerate(colvar_data_all):
            axes[n].plot(data.iloc[:, 0], data.iloc[:, n + 1], label=f'Bias Potential Replica {j + 1}')
        axes[n].set_xlabel("Time Step (fs)")
        axes[n].set_ylabel("Bias Potential")
        axes[n].legend()
        axes[n].grid(True)

        figs.suptitle('CVs and Bias Over Time', fontsize=20)
        figs.savefig("CV_trajectory.png", bbox_inches="tight")
        plt.close()

