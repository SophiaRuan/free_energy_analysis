import pandas as pd
import numpy as np
from free_energy_analysis.free_energy_tool import EnergyCorrectionAnalyzer

def test_correct_free_energy_single():
    df = pd.DataFrame({
        'formula': ['H8LiO4', 'H6LiO3Cl', 'H12Li2O6Cl'],
        'energy': [0.5, 0.6, 0.8]
    })

    x_free = [0.85]  # Single value list for x_free
    conc = 2
    solubility = 20

    df_corrected = EnergyCorrectionAnalyzer.correct_free_energy_multi_solvent(
        df,
        corrections=[{
            'x_free': x_free[-1],
            'activity': EnergyCorrectionAnalyzer.default_activity_model(conc),
            'element': 'O'
        }]
    )

    assert 'energy_corrected' in df_corrected.columns
    assert 'probability_normalized' in df_corrected.columns
    assert 'energy_normalized' in df_corrected.columns
    assert np.isclose(df_corrected['probability_normalized'].sum(), 1.0), "Probabilities not normalized"
    print("Single-solvent correction passed.")

    df_corrected.to_csv("df_corrected_test.csv")
    EnergyCorrectionAnalyzer.plot_corrected_free_energy(df_corrected, "test")

def test_correct_free_energy_multi():
    df = pd.DataFrame({
        'formula': ['H8LiCO4', 'H6LiC2O3', 'H10Li2CO5Cl'],
        'energy': [1.0, 1.2, 1.4]
    })

    x_free_O = 0.35
    x_free_C = 0.20
    activity_O = EnergyCorrectionAnalyzer.default_activity_model(5)
    activity_C = 0.6  # Assume constant

    df_corrected = EnergyCorrectionAnalyzer.correct_free_energy_multi_solvent(
        df,
        corrections=[
            {'x_free': x_free_O, 'activity': activity_O, 'element': 'O'},
            {'x_free': x_free_C, 'activity': activity_C, 'element': 'C'}
        ]
    )

    assert 'energy_corrected' in df_corrected.columns
    assert 'probability_normalized' in df_corrected.columns
    assert 'energy_normalized' in df_corrected.columns
    assert np.isclose(df_corrected['probability_normalized'].sum(), 1.0), "Probabilities not normalized"
    print("Multi-solvent correction passed.")

    # save dataframe
    df_corrected.to_csv("df_corrected_test_mutiple.csv")
    EnergyCorrectionAnalyzer.plot_corrected_free_energy(df_corrected, "test_multiple")




if __name__ == '__main__':
    test_correct_free_energy_single()
    test_correct_free_energy_multi()

