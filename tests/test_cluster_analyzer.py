import os
import pickle
import pandas as pd
import numpy as np
from free_energy_analysis.free_energy_tool import ClusterAnalyzer
from free_energy_analysis.free_energy_tool import EnergyCorrectionAnalyzer
import matplotlib.pyplot as plt


def test_cluster_analyzer():
    # Mock cluster object for testing
    class MockCluster:
        def __init__(self, formula, timestamp, replica_id, ori_idx):
            self.formula = formula
            self.info = {"timestamp": timestamp, "replica_id": replica_id, "ori_idx": ori_idx}
        def get_chemical_formula(self):
            return self.formula

    class MockObj:
        def __init__(self, clusters):
            self.clusters = clusters
            self.cluster_types = {"LiClO": 2, "LiCl": 1}
# Dummy data setup
    mock_clusters = [
        MockCluster("LiClO", 1000, 0, [1, 2]),
        MockCluster("LiCl", 2000, 1, [3, 4]),
        MockCluster("LiClO", 3000, 0, [5, 6])
    ]



    bias_pot_data = {
        "step": [1000, 2000, 3000],
        "replica_id": [0, 1, 0],
        "E_metadyn_d1": [1.0, 0.5, 1.2]
    }
    bias_pot_df = pd.DataFrame(bias_pot_data)


    # Initialize analyzer
    analyzer = ClusterAnalyzer(
        obj=MockObj(mock_clusters),
        bias_pot_df=bias_pot_df,
        target_atoms_ix=[1, 3],
        formula_list=["LiClO", "LiCl"],
        temperature=298
    )

    # Run tests
    formulas = analyzer.get_all_formulas()
    assert set(formulas) == {"LiClO", "LiCl"}

    replicas_LiClO, timesteps_LiClO, bias_LiClO = analyzer.extract_biased_potential_for_formula("LiClO")
    print(replicas_LiClO, timesteps_LiClO, bias_LiClO)
    assert replicas_LiClO == [0]
    assert timesteps_LiClO == [1000]

    # test total data
    total_data = analyzer.calculate_total_cluster_data()
    assert "clu_type" in total_data.columns

    pickle_file = os.path.join("./", "test_cluster_data.pkl")
    analyzer.save_total_cluster_data("test_cluster_data.pkl", output_directory="./")
    loaded_data = analyzer.load_total_cluster_data(pickle_file)
    assert isinstance(loaded_data, pd.DataFrame)
    print("loaded data is a dataframe")


    formula_data = analyzer.get_formula_data()
    probs_total = formula_data["probability"]
    assert np.isclose(np.sum(probs_total), 1.0, atol=1e-2)
    print("unbiased probability is computed")

    energies = formula_data["energy"]
    assert len(energies) == len(probs_total)
    print("free energy is computed")

    # Prepare correction
    df = pd.DataFrame({
        "formula": ["LiClO", "LiCl"],
        "energy": [0.5, 1.0]
    })
    x_free_list = [0.8]
    df_corrected = EnergyCorrectionAnalyzer.correct_free_energy_multi_solvent(
        df,
            corrections=[{
                'x_free': x_free_list[-1],
                'activity': EnergyCorrectionAnalyzer.default_activity_model(5),
                'element': 'O'
            }]
    )

    # Plot results
    EnergyCorrectionAnalyzer.plot_corrected_free_energy(df_corrected, save_name=os.path.join("./", "corrected_plot"))
    print("Energy correction is finished")

if __name__ == '__main__':
    test_cluster_analyzer()
