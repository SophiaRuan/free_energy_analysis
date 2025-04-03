import MDAnalysis as mda
import MDAnalysis.transformations as trans
import os
import json
import pickle
from tqdm import tqdm

from solvation_analysis.solute import Solute 
from solvation_analysis.residence import Residence

class SolvationAnalyzer:
    def __init__(self, base_data_file, directories, Li_id, radii={"Li+": 5}, step=10):
        """
        Initialize the SolvationAnalyzer.
        
        Parameters:
        - base_data_file: Path to the base LAMMPS data file.
        - directories: List of directories containing trajectory files.
        - Li_id: ID of the Li atom to analyze.
        - radii: Dictionary of radii for the solvation analysis.
        - step: Step interval for processing the trajectories.
        """
        self.base_data_file = base_data_file
        self.directories = directories
        self.Li_id = Li_id
        self.radii = radii
        self.step = step
        self.results = {}

        trajectory_files = [
            os.path.join(dir, f"lammps.298K.prod.{i+1:02}.mtd.lammpsdump")
            for i, dir in enumerate(self.directories)
        ]
        self.trajectory_files = trajectory_files

    def process_trajectories(self):
        """
        Process the trajectories to calculate solvation properties.
        """
        
        residence_time_cutoff_list = []
        residence_time_fit_list = []
        radii_list = []
        local_residence_time_list = []

        for i, traj in tqdm(enumerate(self.trajectory_files), desc="Processing trajectories"):
            u = mda.Universe(self.base_data_file, traj)
            ag = u.atoms
            transform = mda.transformations.wrap(ag)
            u.trajectory.add_transformations(transform)

            # Atom selection
            O = u.select_atoms("type 1")
            Li = u.select_atoms("type 3")
            Cl = u.select_atoms("type 4")
            H = u.select_atoms("type 2")
            li = u.select_atoms(f"id {self.Li_id}")

            # Solvation analysis
            solute_li = Solute.from_atoms(
                li, {"H2O_O": O, "H2O_H": H, "Cl-": Cl, "Li+": Li - li},
                solute_name="Li+", radii=self.radii
            )
            solute_li.run(step=self.step)
            residence = Residence.from_solute(solute_li)

            # Collect results
            residence_time_cutoff_list.append(residence.residence_times_cutoff)
            residence_time_fit_list.append(residence.residence_times_fit)
            radii_list.append(solute_li.radii)

            # Calculate local residence time
            Cl_res_cutoff = residence.residence_times_cutoff.get("Cl-", 0)
            O_res_cutoff = residence.residence_times_cutoff.get("H2O_O", 0)
            Cl_res_fit = residence.residence_times_fit.get("Cl-", 0)
            O_res_fit = residence.residence_times_fit.get("H2O_O", 0)

            cl_res_time = min(Cl_res_cutoff, Cl_res_fit) if Cl_res_cutoff or Cl_res_fit else 0
            o_res_time = min(O_res_cutoff, O_res_fit) if O_res_cutoff or O_res_fit else 0
            local_residence_time_list.append(max(cl_res_time, o_res_time))

        local_residence_time = round(min(local_residence_time_list))
        
        self.results = {
            "local_residence_time": local_residence_time,
            "residence_time_cutoff_list": residence_time_cutoff_list,
            "residence_time_fit_list": residence_time_fit_list,
            "radii_list": radii_list,
        }
        return self.results

    def calculate_average_radii(self):
        """
        Calculate the average radii from a single trajectory.
        """
        u = mda.Universe(self.base_data_file, self.trajectory_files)
        O = u.select_atoms("type 1")
        Li = u.select_atoms("type 3")
        Cl = u.select_atoms("type 4")
        H = u.select_atoms("type 2")

        solute_li = Solute.from_atoms(
            Li, {"H2O_O": O, "Cl-": Cl, "H2O_H": H}, solute_name="Li+"
        )
        solute_li.run(step=self.step)
        self.results["average_radii"] = solute_li.radii
        self.results["average_CN"] = solute_li.coordination.coordination_numbers
        
        return self.results["average_radii"]

    def save_results(self, filename="solvation_results.pkl"):
        """
        Save the results to a pickle file.
        """
        with open(filename, "wb") as pickle_file:
            pickle.dump(self.results, pickle_file)

    def save_results_json(self, filename="solvation_results.json"):
        """
        Save the results to a JSON file.
        """
        with open(filename, "w") as json_file:
            json.dump(self.results, json_file, indent=4)

    def load_results(self, filename):
        """
        Load results from a pickle file.
        """
        with open(filename, "rb") as pickle_file:
            self.results = pickle.load(pickle_file)
        return self.results

