from sea_urchin.sea_urchin import SeaUrchin
from sea_urchin.plotting.rendering import plot_structures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

import os
import glob
import pickle
import re

def get_multiple_replica_files(base_path):
    # Regex patterns to extract the number from the filenames
    # partial_pattern = re.compile(r'colvar\.out\.partial\.(\d+)\.pmf')
    # full_pattern = re.compile(r'colvar\.out\.(\d+)\.pmf')
    trj_files = []
    partial_pmf_files = []
    bias_potential_files = []
    full_pmf_files=[]

    dir_path_list = glob.glob(os.path.join(base_path, '*_IDNR'))
    dir_path_sorted = sorted(dir_path_list)
    # Iterate through each *_IDNR directory
    for dir_path in dir_path_sorted:
        # Separate matching files into partial and full categories
        trj = glob.glob(os.path.join(dir_path, "lammps.*K.prod.*.mtd.lammp*"))
        trj = sorted(trj)
        bias_potential_file = glob.glob(os.path.join(dir_path, "colvar.out.colvars.traj"))
        partial_pmf_file = glob.glob(os.path.join(dir_path, 'colvar.out.partial.pmf'))
        full_pmf_file = glob.glob(os.path.join(dir_path, 'colvar.out.pmf'))


        trj_files.append(trj)
        bias_potential_files.append(bias_potential_file)
        partial_pmf_files.append(partial_pmf_file)
        full_pmf_files.append(full_pmf_file)
    # Find the latest file for both categories
    return dir_path_sorted, trj_files, bias_potential_files, partial_pmf_files, full_pmf_files

def load_bias_potential_data(bias_potential_files):
    """Load bias potential data from multiple files into a single DataFrame."""
    bias_pot_df = pd.DataFrame()
    df = pd.read_csv(bias_potential_files[0][0], sep="\s+", comment="#")
    if len(df.columns)==4:
        col_names = ["step", "coord1", "coord2", "bias_potential"]
    else:
        col_names = ["step", "coord1", "coord2", "coord3", "bias_potential"]

    for i, file in enumerate(bias_potential_files):
        df = pd.read_csv(file[0], sep="\s+", comment="#", names=col_names)
        df["replica_id"]=i
        bias_pot_df = pd.concat([bias_pot_df, df], ignore_index=True)
    return bias_pot_df



class ClusterAnalyzer:
    def __init__(self,
                 obj,
                 bias_pot_df,
                 target_atoms_ix=None,
                 formula_list=None,
                 num_formulas=None,
                 geometry_pickle_data=None,
                 temperature=298):
        """
        Initialize the ClusterAnalysis class.

        Parameters:
        - obj: An object containing the cluster data from sea_urchin.
        - bias_pot_df: A DataFrame containing the bias potential data.
        - : all the cluster data including the replica, timestep and biased potential applied
        - target_atom_ix: If provided a list of index of atom, the clusters including these atoms will be analyzed. e.g. When we control 1 Li+ in metadynamics.
        - geometry_pickle_data: Pickle data for geometry-level cluster analysis.
        - formula_list: A list of formulas to analyze.
        - num_formulas: Number of formulas to analyze (if no list is provided).
        - temperature: Temperature of the system.
        """

        self.obj = obj
        self.bias_pot_df = bias_pot_df
        self.target_atoms_ix = target_atoms_ix
        self.formula_list = formula_list
        self.geometry_pickle_data = geometry_pickle_data
        self.temperature = temperature
        self.num_formulas = num_formulas
        self.k_b = 1.380649e-23  # Boltzmann constant (in J)
        self.kbT = self.k_b * self.temperature
        self.convert_kcal_per_mol_J = 4184 / 6.02e23
        self.convert_kcal_per_mol_kbT = self.convert_kcal_per_mol_J / self.kbT
        if self.target_atoms_ix:
            target_atoms_clusters = []
            for clu in obj.clusters:
                if any(idx in clu.info["ori_idx"] for idx in self.target_atoms_ix):
                    target_atoms_clusters.append(clu)
        self.clusters = obj.clusters if target_atoms_ix is None else target_atoms_clusters

    def calculate_total_cluster_data(self):
        """Calculate the total_cluster_data by extracting bias potentials for all formulas."""
        total_cluster_data = {
            "clu_type": [],
            "replica": [],
            "timestep": [],
            "bias_potential": []
        }

        formulas = self.get_all_formulas()

        for formula in formulas:
            replica_id_list, timestep_list, bias_potential_list = self.extract_biased_potential_for_formula(formula)
            total_cluster_data["clu_type"].append(formula)
            total_cluster_data["replica"].append(replica_id_list)
            total_cluster_data["timestep"].append(timestep_list)
            total_cluster_data["bias_potential"].append(bias_potential_list)

        self.total_cluster_data = pd.DataFrame(total_cluster_data)
        return self.total_cluster_data

    def save_total_cluster_data(self, filename, output_directory="./"):
        """Save the total_cluster_data to a pickle file."""
        path = os.path.join(output_directory, filename)
        with open(path, 'wb') as f:
            pickle.dump(self.total_cluster_data, f)
        print(f"Total cluster data saved to {path}.")

    def load_total_cluster_data(self, path_to_file):
        """Load the total_cluster_data from a pickle file."""
        with open(path_to_file, 'rb') as f:
            self.total_cluster_data = pickle.load(f)
        print(f"Total cluster data loaded from {path_to_file}.")
        return self.total_cluster_data

    ### Method to get all formulas ###

    def get_all_formulas(self):
        """Return all possible formulas from the object."""
        return list(set([cluster.get_chemical_formula() for cluster in self.clusters]))

    ### Case1: Geometry-level Analysis ###
    def get_geometry_data(self):
        """Perform geometry-level analysis if the geometry pickle data is provided."""

        def extract_biased_potential_for_geometry(geometry_label):
            """Extract bias potential for a specific geometry from the bias potential DataFrame."""
            replica_id_list, timestep_list, bias_potential_list = [], [], []
            clusters = [clu for idx, clu in enumerate(geometry_pickle_data.clusters) if geometry_pickle_data.labels[idx] == geometry_label]
            formula = clusters[0].get_chemical_formula()

            for idx, clu in enumerate(clusters):
                timestep = clu.info["timestamp"]
                replica_id = clu.info["replica_id"]

                if timestep is not None and replica_id is not None:
                    timestep_list.append(timestep)
                    replica_id_list.append(replica_id)

                    bias_potential = self.get_bias_potential_for_step(replica_id, timestep, idx, formula)
                    bias_potential_list.append(bias_potential)
            return replica_id_list, timestep_list, bias_potential_list

        if self.geometry_pickle_data is None:
            raise ValueError("Geometry pickle data is not provided.")

        geometry_data = {"geometry_label": [],
                         "replica": [],
                         "timestep": [],
                         "bias_potential": [],
                         "probability": [],
                         "energy": []}
        geometry_labels = list(set(self.geometry_pickle_data.labels))

        for geometry_label in geometry_labels:
            replica_id_list, timestep_list, bias_potential_list = extract_biased_potential_for_geometry(geometry_label)
            probability_list = self.compute_probability(bias_potential_list)
            energy_list = self.get_relative_energy_from_prob(probability_list)

            geometry_data["geometry_label"].append(geometry_label)
            geometry_data["replica"].append(replica_id_list)
            geometry_data["timestep"].append(timestep_list)
            geometry_data["bias_potential"].append(bias_potential_list)
            geometry_data["probability"].append(probability_list)
            geometry_data["energy"].append(energy_list)

            with open(os.path.join(output_directory, geometry_data.pkl), 'wb') as f:
                pickle.dump(geometry_data, f)
            print(f"Total cluster data saved to geometry_data.pkl.")

        return pd.DataFrame(geometry_data)



    ### Case 2: Formula-level Analysis ###

    def get_formula_data(self, output_directory="./"):
        """Perform formula-level analysis if only formula list or formula number is provided."""
        if self.formula_list is None and self.num_formulas is None and self.target_atoms_ix is None:
            raise ValueError("Neither a formula list nor a number of formulas is provided.")

        formula_data = {"formula": [], "replica": [], "timestep": [], "bias_potential": [], "probability": [], "energy": []}

        if self.formula_list is not None:
            desired_formulas = self.formula_list
        elif self.target_atoms_ix is not None:
            desired_formulas = self.get_all_formulas()
        else:
            cluster_sorted = sorted(self.obj.cluster_types.items(), key=lambda item: item[1], reverse=True)
            desired_formulas = list(dict(cluster_sorted).keys())[:self.num_formulas]

        for formula in desired_formulas:
            replica_id_list, timestep_list, bias_potential_list = self.extract_biased_potential_for_formula(formula)
            probability_list = self.compute_probability(bias_potential_list)
            energy_list = self.get_relative_energy_from_prob(probability_list)

            formula_data["formula"].append(formula)
            formula_data["replica"].append(replica_id_list)
            formula_data["timestep"].append(timestep_list)
            formula_data["bias_potential"].append(bias_potential_list)
            formula_data["probability"].append(probability_list)
            formula_data["energy"].append(energy_list)

            with open(os.path.join(output_directory, "formula_data.pkl"), 'wb') as f:
                pickle.dump(formula_data, f)
            print(f"Total cluster data saved to formula_data.pkl.")
        return pd.DataFrame(formula_data)


    def extract_biased_potential_for_formula(self, formula):
        """Extract bias potential for a specific formula from the bias potential DataFrame."""
        replica_id_list, timestep_list, bias_potential_list = [], [], []
        for idx, cluster in enumerate(self.clusters):
            if cluster.get_chemical_formula() == formula:
                timestep = cluster.info["timestamp"]
                replica_id = cluster.info["replica_id"]

                if timestep is not None and replica_id is not None:
                    timestep_list.append(timestep)
                    replica_id_list.append(replica_id)

                    bias_potential = self.get_bias_potential_for_step(replica_id, timestep, idx, formula)
                    bias_potential_list.append(bias_potential)

        return replica_id_list, timestep_list, bias_potential_list

    ### Common Methods for Both Cases ###

    def compute_probability(self, bias_potential_list):
        count_total_clusters = len(self.clusters)
        # print(max(bias_potential_list))
        biased_probabilities = np.ones(len(bias_potential_list)) / count_total_clusters
        # Apply bias potential correction
        bias_potential = np.array(bias_potential_list) * self.convert_kcal_per_mol_kbT # Convert from kcal/mol to kbT
        logsumexp_trick = bias_potential+np.log(biased_probabilities) # q = p*e^(bias_potential/kbT), P = sum(q)/Q

        total_sum_bias = self.get_total_sum_bias()

        return np.exp(logsumexp(logsumexp_trick)-total_sum_bias)


    def get_total_sum_bias(self):
        # total_sum_bias = 0
        logsumexp_trick_list = []
        count_total_clusters = len(self.clusters)

        for idx, formula in enumerate(self.total_cluster_data["clu_type"]):
            bias_potential_list = np.array(self.total_cluster_data.iloc[idx]["bias_potential"])* self.convert_kcal_per_mol_kbT # convert to J/mol
            biased_probabilities = np.ones(len(bias_potential_list)) / count_total_clusters
            bias_potential = np.array(bias_potential_list)  # Convert from kcal/mol to kbT
            logsumexp_trick = bias_potential+np.log(biased_probabilities)
            logsumexp_trick_list.append(logsumexp_trick)
        return logsumexp(np.concatenate(logsumexp_trick_list)) # return a logsumexp value = log(sum(P*e^(beta*bias_potential)))


    def get_bias_potential_for_step(self, replica_id, timestep, cluster_idx, label):
        """Get bias potential for a specific replica and timestep."""
        try:
            bias_potential = self.bias_pot_df[
                (self.bias_pot_df["step"] == timestep) &
                (self.bias_pot_df["replica_id"] == replica_id)
            ]["E_metadyn_d1"].values[0]
        except (IndexError, KeyError):
            print(f"No bias potential recorded for {label} at replica {replica_id}, timestep {timestep}")
            bias_potential = 0
        return bias_potential

    def get_relative_energy_from_prob(self, prob):
        """Calculate relative free energy from probabilities."""
        return -np.log(prob)

def logsumexp(x):
    # first reduce max value c among all number
    # take exponential, sum and log,
    # and eventually add c to all value
    c = x.max()
    return c + np.log(np.sum(np.exp(x-c)))


class EnergyCorrectionAnalyzer:
    def __init__(
        self,
        base_path: str,
        nstrides: int,
        data_file: str,
        traj_list: list[str],
        T: float = 298,
        activity_model: callable = None,
        activity_table: list[tuple[float, float]] = None,
        central_atom_id: int = None,
        solute_atoms_per_molecule: int = 2,
        solute_selector: callable = None,
        non_free_selector: callable = None,
        all_solvents_selector: callable = None
    ):
        """
        Initialize the analyzer with simulation parameters.

        Parameters:
        - base_path: base directory of data
        - nstrides: stride interval for reading trajectory
        - data_file: cluster data file
        - traj_list: list of MDAnalysis.Universe object paths or labels
        - T: temperature in K
        - activity_model: optional user-defined function, activity = f(conc)
        Default activity model is for water in aqueous LiCl at 298 K (if none is provided):
            activity = -0.0444 * conc + 1.0014
            Example
            def linear_activity(conc: float, solubility) -> float:
                def get_activity(conc):
                    return -0.0444*conc + 1.0014 # 298K water activity in LiCl aqueous solution
                    return -0.0507*conc + 1 # 283K water activity in LiCl aqueous solution
                    return -0.0422*conc + 1 # 313K water activity in LiCl aqueous solution

                activity = get_activity(conc) if conc <= solubility else get_activity(solubility)
                return activity

        - activity_table: optional table for activity vs. concentration (will be interpolated)
        - central_atom_id: ID of ion (e.g., Li+) around which to define local environment
        - solute_atoms_per_molecule: number of atoms per solute molecule (default binary salt = 2)
        - solute_selector: selects solute atoms around center
        - non_free_selector: selects target bound solvent residues
        - all_solvents_selector: selects all solvent residues around center
        Selector functions must take the form:
            def selector(universe, central_atom_id: int, distance: float) -> AtomGroup:
                return universe.select_atoms("your selection string")

        Example selectors for aqueous LiCl system:

            # select all the solute species around center species
            def solute_selector(u, center_species_id, distance):
                # 3 and 4 are Li and Cl
                return u.select_atoms(f"byres (around {d} (id {center_species_id}) and (type 3 or type 4))")

            # select all the non free water around center species
            def non_free_selector(u, center_species_id, distance):
                # O around Li and H around Cl are both considered bound (not free) water!!!
                return u.select_atoms(
                    f"byres ((around {distance} (id {center_species_id})) and ((type 1 and around 2.65 (type 3)) or (type 2 and around 2.95 (type 4))))"
                )

            # select all solvents around center species (there can be solvents other than water)
            def all_solvents_selector(u, center_species_id, distance):
                return u.select_atoms(f"byres ((type 1 or type 2) and (around {distance} (id {center_species_id})))}")
        """

        self.base_path = base_path
        self.nstrides = nstrides
        self.data_file = data_file
        self.traj_list = traj_list
        self.T = T
        self.central_atom_id = central_atom_id
        self.solute_atoms_per_molecule: int = solute_atoms_per_molecule
        self.solute_selector = solute_selector
        self.non_free_selector = non_free_selector
        self.all_solvents_selector = all_solvents_selector

        # If activity table is given, interpolate it
        if activity_model:
            self.activity_model = activity_model
        elif activity_table:
            conc_vals, activity_vals = zip(*activity_table)
            self.activity_model = interp1d(conc_vals, activity_vals, kind='linear', fill_value='extrapolate')
         else:
            self.activity_model = EnergyCorrectionAnalyzer.default_activity_model

    @staticmethod
    def default_activity_model(conc: float) -> float:
        """Default linear activity model for water at 298 K."""
        return max(-0.0444 * conc + 1.0014, 0.01)

    @staticmethod
    def count_element(formula, element="O"):
        """Count the number of atoms in a chemical formula."""
        matches = re.findall(fr'{element}(\d*)', formula)
        return sum(int(m) if m else 1 for m in matches)

    def calculate_free_solvent_fraction(self, u_list: list, distance_range: range = range(12, 13)) -> list[float]:
        x_free_list = []

        for x in tqdm(distance_range):
            x_free_means = []
            for u in u_list:
                x_frees = []

                for ts in u.trajectory[::20]:
                    if not self.solute_selector or not self.non_free_selector or not self.all_solvents_selector:
                        raise ValueError(
                            "All selector functions (solute, non-free, all_solvents) must be provided.")

                    num_all_solutes = len(self.solute_selector(u, self.central_atom_id, x)) / self.solute_atoms_per_molecule
                    num_non_free_solvent = len(set(self.non_free_selector(u, self.central_atom_id, x).residues))
                    num_all_solvents = len(set(self.all_solvents_selector(u, self.central_atom_id, x).residues))

                    try:
                        x_free = (num_all_solvents - num_non_free_solvent) / (num_all_solvents + num_all_solutes)
                        x_frees.append(x_free)
                    except ZeroDivisionError:
                        continue

                if x_frees:
                    x_free_means.append(np.mean(x_frees))

            x_free_list.append(np.mean(x_free_means))
        return x_free_list

    def get_activity_from_conc(self, conc: float, solubility: float) -> float:
        """Return activity coefficient from model or table."""
        if not self.activity_model:
            raise ValueError("No activity model or table provided.")

        return float(self.activity_model(conc)) if conc <= solubility else float(self.activity_model(solubility))

    # @staticmethod
    # def correct_free_energy(
    #     df: pd.DataFrame,
    #     x_free_list: list[float],
    #     conc: float,
    #     solubility: float,
    #     element: str = 'O',
    #     activity_model: callable = None
    # ) -> pd.DataFrame:
    #     """
    #     Apply correction using a single solvent type.
    #
    #     Parameters:
    #     - df: input DataFrame
    #     - x_free_list: list of free solvent fractions
    #     - conc: solute concentration
    #     - solubility: solubility limit
    #     - element: atom type to count (e.g., 'O')
    #     - activity_model: optional activity model callable (default water model used if None)
    #     """
    #     x_bulk = 1.0
    #     activity_model = activity_model or EnergyCorrectionAnalyzer.default_activity_model
    #     activity = activity_model(conc) if conc <= solubility else activity_model(solubility)
    #     delta_mu = np.log(x_free_list[-1] * activity / x_bulk)
    #
    #     df = df.copy()
    #     df["N_target_atoms"] = [EnergyCorrectionAnalyzer.count_element(f, element=element) for f in df["formula"]]
    #     df["energy_corrected"] = df["energy"] - df["N_target_atoms"] * delta_mu
    #     df["probability_normalized"] = np.exp(-df["energy_corrected"])
    #     df["probability_normalized"] /= np.sum(df["probability_normalized"])
    #     df["energy_normalized"] = -np.log(df["probability_normalized"])
    #     return df.sort_values("energy_normalized")

    @staticmethod
    def correct_free_energy_multi_solvent(df: pd.DataFrame, corrections: list[dict]) -> pd.DataFrame:
        """
        Apply one or multiple solvent-based corrections.

        Parameters:
        - df: input DataFrame
        - corrections: list of dicts, each with:
            - 'x_free': float
            - 'activity': float
            - 'element': str

        Example usage:
        analyzer_O = EnergyCorrectionAnalyzer(...)
        x_free_1 = analyzer_O.calculate_free_solvent_fraction([...])
        df_corrected = EnergyCorrectionAnalyzer.correct_free_energy_multi_solvent(df, [
        {'x_free': x_free_1[-1], 'activity': analyzer_O.get_activity_from_conc(conc=10, solubility=20), 'element': 'O'}
        ])

        """
        df = df.copy()
        total_delta_mu = np.zeros(len(df))

        for corr in corrections:
            x_free = corr['x_free']
            activity = corr['activity']
            element = corr['element']
            delta_mu = np.log(x_free * activity)
            n_atoms = np.array([EnergyCorrectionAnalyzer.count_element(f, element) for f in df["formula"]])
            total_delta_mu += n_atoms * delta_mu

        df["energy_corrected"] = df["energy"] - total_delta_mu
        df["probability_normalized"] = np.exp(-df["energy_corrected"])
        df["probability_normalized"] /= np.sum(df["probability_normalized"])
        df["energy_normalized"] = -np.log(df["probability_normalized"])
        return df.sort_values("energy_normalized")

    def plot_corrected_free_energy(self, df_corrected_sorted):
        top = df_corrected_sorted.head(5)

        plt.plot(top["formula"], top["energy_normalized"])
        plt.ylabel("Corrected Free Energy (kbT)")
        plt.xlabel("Formula")
        plt.savefig("corrected_free_energy.png", bbox_inches="tight")
        plt.close()

        plt.plot(top["formula"], top["probability_normalized"])
        plt.ylabel("Corrected Probability")
        plt.xlabel("Formula")
        plt.savefig("corrected_probability.png", bbox_inches="tight")
        plt.close()


