from message_ix_models import Context
from message_ix_models.model.water.cli import nexus, modify_scenario

# Configuration
BASE_URL = "ixmp://ixmp_dev/MESSAGE_GLOBIOM_SSP2_v6.1/baseline"
REGIONS = "R12"
SSP = "SSP2"

# Build mode: choose 'full' or 'modify'
BUILD_MODE = "modify"  # Change to 'full' for complete rebuild

# Scenario parameters
RCP_ORIGINAL = "7p0"  # RCP of existing scenario (if modifying)
RCP_NEW = "2p6"  # Target RCP
SDG = "baseline"
REL = "high"

if BUILD_MODE == "full":
    # Full build approach - creates new scenario from scratch
    print("🔨 Building new nexus scenario from scratch...")

    ctx = Context()
    ctx.handle_cli_args(url=BASE_URL)
    ctx.ssp = SSP

    # Use the CLI function - handles all initialization, building, and solving
    nexus(ctx, regions=REGIONS, rcps=RCP_NEW, sdgs=SDG, rels=REL, macro=False)

    print("✅ Full build completed!")

elif BUILD_MODE == "modify":
    # Efficient modify approach - updates existing scenario
    print("Modifying existing nexus scenario...")

    # First, check if we have an existing nexus scenario to modify
    existing_url = f"ixmp://ixmp_dev/MESSAGE_GLOBIOM_SSP2_v6.1/baseline_nexus_7_high"

    try:
        ctx = Context()
        ctx.handle_cli_args(url=existing_url)
        ctx.ssp = SSP

        print(
            f"📝 Found existing scenario, modifying RCP from {RCP_ORIGINAL} to {RCP_NEW}"
        )

        # Use the new modify CLI function for efficient parameter changes
        modify_scenario(ctx, regions=REGIONS, rcps=RCP_NEW)

        print("✅ Efficient modification completed!")

    except Exception as e:
        print(f"❌ Could not modify existing scenario: {e}")
        print("🔄 Falling back to full build...")

        # Fallback to full build if modification fails
        ctx = Context()
        ctx.handle_cli_args(url=BASE_URL)
        ctx.ssp = SSP

        nexus(ctx, regions=REGIONS, rcps=RCP_NEW, sdgs=SDG, rels=REL, macro=False)

        print("✅ Fallback build completed!")

else:
    raise ValueError(f"Invalid BUILD_MODE: {BUILD_MODE}. Must be 'full' or 'modify'")

print(f"🎉 Script completed successfully with {BUILD_MODE} approach!")
