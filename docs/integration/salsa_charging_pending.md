# Salsa Charging Pending Integration

This config is not active yet.

Current `main_future` does not appear to include the required charging mechanism. The parked config is stored at:

```text
data/input/charging/salsa_charging_config.json
```

Required mechanism touchpoints include `model/robot.py`, `engine/universe.py`, and NetLogo setup/bridge charger overlay logic.

Do not enable this config until the mechanism is implemented and validated.

## Source Branch Material

The pending config and notes were ported from Salsa's `salsa_charging_baseline` branch:

- `final_solution/charging_config.json`
- `final_solution/README.md`
- `final_solution/CODE_TOUCHPOINTS.md`

The branch documentation describes a hybrid charger placement and policy configuration, but those settings are only meaningful after the simulation has active charging state, dispatch, charger occupancy, and NetLogo overlay support.
