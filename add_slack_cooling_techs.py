#!/usr/bin/env python3
"""Add slack to growth_activity_up and initial_activity_up for cooling technologies."""

from ixmp import Platform
from message_ix import Scenario

# Technologies to modify
TECHNOLOGIES = [
    "geo_ppl__ot_saline",
    "geo_ppl",
    "igcc__ot_saline",
    "igcc",
    "coal_adv__ot_saline",
    "nuc_hc__ot_saline",
    "gas_cc__ot_saline",
    "igcc_ccs__ot_saline",
    "coal_adv_ccs__ot_saline",
    "bio_istig__ot_saline",
]

# Parameters to modify
PARAMETERS = ["growth_activity_up", "initial_activity_up"]

# Slack factor (50% increase)
SLACK_FACTOR = 0.5

# Initialize platform
mp = Platform(name="ixmp_dev", jvmargs=["-Xmx20G"])

# Load the base scenario (same as electricity removal script)
source_scen = Scenario(
    mp, model="clone_geidco_test_SSP2_v5.3", scenario="baseline_geidco_test_nexus_gamma"
)

# Clone to create new scenario
new_scen = source_scen.clone(
    model="clone_geidco_test_SSP2_v5.3",
    scenario="nexus_cooling_tech_slack_50pct",
    keep_solution=False,
)


# Process each parameter
for param in PARAMETERS:
    with new_scen.transact(
        f"Add {SLACK_FACTOR * 100}% slack to {param} for cooling technologies in 2030"
    ):
        # Get existing data for the parameter, filtering for our technologies and year 2030
        if param == "growth_activity_up":
            # growth_activity_up has year_act dimension
            existing_data = new_scen.par(
                param, filters={"technology": TECHNOLOGIES, "year_act": 2030}
            )
        else:
            # initial_activity_up has year_act dimension as well
            existing_data = new_scen.par(
                param, filters={"technology": TECHNOLOGIES, "year_act": 2030}
            )

        if existing_data.empty:
            print(f"\nNo existing data for {param} in 2030")
            continue

        print(
            f"\nProcessing {param}: Found {len(existing_data)} rows for specified technologies in 2030"
        )

        # Create a copy to modify
        tech_data = existing_data.copy()

        # Store original values for reporting
        original_values = tech_data.groupby("technology")["value"].mean()

        # Add slack by multiplying values by (1 + slack_factor)
        tech_data["value"] = tech_data["value"] * (1 + SLACK_FACTOR)

        # Update the parameter by adding the modified data
        # MESSAGEix will handle the update internally
        new_scen.add_par(param, tech_data)

        # Report changes
        for tech in TECHNOLOGIES:
            tech_rows = len(tech_data[tech_data["technology"] == tech])
            if tech_rows > 0:
                orig_val = original_values.get(tech, 0)
                new_val = orig_val * (1 + SLACK_FACTOR)
                print(
                    f"  {tech}: {tech_rows} rows modified ({orig_val:.3f} -> {new_val:.3f})"
                )

print("\nSlack addition complete!")

# Solve the scenario
print("\nSolving scenario...")
try:
    new_scen.solve(solve_options={"lpmethod": "4"})
    print("Scenario solved successfully!")
except Exception as e:
    print(f"Error solving scenario: {e}")
