import ixmp as ix
import message_ix


mp = ix.Platform(name="ixmp_dev", jvmargs=["-Xmx14G"])


# Source scenario details
source_model = "clone_SSP_SSP5_v4.0"
source_scenario = "baseline_clone_testing"


# Target scenario details
# load scenario, but DO NOT MAKE CHANGES to it
sc_ref = message_ix.Scenario(mp, source_model, source_scenario, cache=True)
# clone to a scenario you want to use and do what you want
scen = sc_ref.clone(
    model="RLL4_SSP_SSP5_v4.0", scenario="RLL4_clone_testing", keep_solution=False
)
#DSL2 water supply and demand refactor rest legacy
#DSL3 DSL2 + Infra + Agri (water for ppl legacy)
#DSL4 DSL3 + water for ppl 



# mix-models --url="ixmp://ixmp_dev/RLL4_SSP_SSP5_v4.0/RLL4_clone_testing" water-ix --regions=R12 cooling --ssp=SSP2
