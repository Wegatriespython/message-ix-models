#!/usr/bin/env python3
"""
Script to remove electricity inputs from all water technologies in a MESSAGEix scenario.
Based on modify_and_solve_scenario.py but targeting water technology electricity inputs.
"""

import yaml
from ixmp import Platform
from message_ix import Scenario


def load_water_technologies_with_electricity():
    """Load the list of water technologies that have electricity inputs."""
    with open("water_technologies_with_electricity.yaml", "r") as f:
        data = yaml.safe_load(f)
    return data["water_technologies_with_electricity"]["complete_list"]


def remove_electricity_inputs_from_water_techs(scenario, water_tech_list):
    """
    Remove electricity inputs from water technologies in the scenario.
    
    Args:
        scenario: MESSAGEix scenario object
        water_tech_list: List of water technology names that use electricity
    """
    
    removed_count = 0
    
    with scenario.transact("Remove electricity inputs from water technologies"):
        for tech in water_tech_list:
            try:
                # Get electricity input data for this technology
                df = scenario.par("input", filters={
                    "technology": tech, 
                    "commodity": "electr"
                })
                
                if not df.empty:
                    # Remove the electricity input parameters
                    # Drop value and unit columns as required by remove_par
                    df_to_remove = df.drop(columns=["value", "unit"])
                    scenario.remove_par("input", df_to_remove)
                    removed_count += len(df)
                    print(f"Removed {len(df)} electricity input entries for technology: {tech}")
                
            except Exception as e:
                print(f"Warning: Could not process technology {tech}: {e}")
                continue
    
    print(f"Total electricity input entries removed: {removed_count}")
    return removed_count


def main():
    """Main function to execute the electricity input removal process."""
    
    # Load list of water technologies with electricity inputs
    try:
        water_tech_list = load_water_technologies_with_electricity()
        print(f"Loaded {len(water_tech_list)} water technologies with electricity inputs")
    except FileNotFoundError:
        print("Error: water_technologies_with_electricity.yaml not found")
        print("Please run extract_water_elec_techs.py first")
        return
    
    # Initialize platform and scenario
    mp = Platform(name="ixmp_dev", jvmargs=["-Xmx20G"])
    
    # Load source scenario
    source_scen = Scenario(
        mp, 
        model="clone_geidco_test_SSP2_v5.3", 
        scenario="baseline_geidco_test_nexus_gamma"
    )
    
    print(f"Loaded source scenario: {source_scen.model}/{source_scen.scenario}")
    
    # Unlock and clone scenario
    source_scen.commit("unlocking for water electricity removal")
    source_scen.check_out()
    
    scen = source_scen.clone(
        model="clone_geidco_test_SSP2_v5.3",
        scenario="Nexus_no_water_elec_inputs",
        keep_solution=False,
    )
    
    print(f"Created new scenario: {scen.model}/{scen.scenario}")
    
    # Remove electricity inputs from water technologies
    print("\nRemoving electricity inputs from water technologies...")
    
    removed_count = remove_electricity_inputs_from_water_techs(scen, water_tech_list)
    
    if removed_count > 0:
        scen.commit(f"Removed electricity inputs from {len(water_tech_list)} water technologies")
        print(f"Successfully removed {removed_count} electricity input entries")
        
        # Solve the scenario
        print("\nSolving modified scenario...")
        try:
            scen.solve()
            print("Scenario solved successfully!")
        except Exception as e:
            print(f"Warning: Scenario solve failed: {e}")
            print("The scenario has been saved but may need manual adjustment before solving")
    else:
        print("No electricity inputs found to remove")
        scen.discard_changes()


if __name__ == "__main__":
    main()