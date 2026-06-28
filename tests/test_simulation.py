import matplotlib
matplotlib.use('Agg')
import numpy as np
from src.core.config import SimulationConfig
from src.core.simulation import Simulation
from src.equations import AdvectionEquation
from src.operators import upwind
from src.integrators import euler
from src.boundaries import periodic
from src.init_conditions import advec_gauss, make_ic

def test_simulation_runs():
    config = SimulationConfig(
        shape=(50,),
        spacing=(0.1,),
        dt=0.01,
        final_time=0.1,
        steps_per_frame=1,
        equation=AdvectionEquation,
        operator=upwind,
        boundary=periodic,
        integrator=euler,
        coefficient=1.0,
        initial_condition=make_ic(advec_gauss)
    )
    
    sim = Simulation(config)
    results = sim.run()
    
    assert results is not None
    assert results.grid.shape == (50,)
    assert results.final_state.data.shape == (1, 50)
    # Just check it ran without error and returned arrays
    assert isinstance(results.final_state.data, np.ndarray)

if __name__ == "__main__":
    test_simulation_runs()
    print("Integration test passed!")
