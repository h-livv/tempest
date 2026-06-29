import matplotlib
matplotlib.use('Agg')
import numpy as np
from src.core.config import SimulationConfig
from src.core.simulation import Simulation
from src.physics.equations import AdvectionEquation
from src.numerics.operators import upwind
from src.numerics.integrators import euler
from src.mesh.boundaries import periodic
from src.physics.init_conditions import GaussianIC

def test_simulation_runs():
    config = SimulationConfig(
        shape=(50,),
        spacing=(0.1,),
        dt=0.01,
        final_time=0.1,
        steps_per_frame=1,
        equation=AdvectionEquation(velocity=1.0),
        operator=upwind,
        boundary=periodic,
        integrator=euler,
        initial_condition=GaussianIC(sigma=10.0, center_ratio=0.5)
    )
    
    sim = Simulation(config)
    results = sim.run()
    
    assert results is not None
    assert results.grid.shape == (50,)
    assert results.final_state.data.shape == (1, 50)
    # Just check it ran without error and returned arrays
    assert isinstance(results.final_state.data, np.ndarray)

def test_2d_shallow_water_rk4():
    from src.physics.equations import ShallowWaterEquation
    from src.numerics.operators import gradient
    from src.numerics.integrators import rk4
    from src.mesh.boundaries import reflect
    from src.physics.init_conditions import ShallowGaussianIC
    
    config = SimulationConfig(
        shape=(10, 10),
        spacing=(0.1, 0.1),
        dt=0.001,
        final_time=0.002,
        steps_per_frame=1,
        equation=ShallowWaterEquation(),
        operator=gradient,
        boundary=reflect,
        integrator=rk4,
        initial_condition=ShallowGaussianIC(center_ratio=0.5, sigma=0.2, ambient_depth=0.5)
    )
    sim = Simulation(config)
    results = sim.run()
    assert results is not None
    assert results.final_state.data.shape == (3, 10, 10)

def test_2d_wave_validation():
    from src.physics.equations import WaveEquation
    from src.numerics.operators import laplacian
    from src.numerics.integrators import leapfrog
    from src.mesh.boundaries import periodic
    
    config = SimulationConfig(
        shape=(10, 10),
        spacing=(0.1, 0.1),
        dt=0.001,
        final_time=0.002,
        steps_per_frame=1,
        equation=WaveEquation(wave_speed=1.0),
        operator=laplacian,
        boundary=periodic,
        integrator=leapfrog,
        initial_condition=GaussianIC(sigma=0.2, num_fields=2)
    )
    sim = Simulation(config)
    results = sim.run()
    assert results is not None
    assert results.final_analytical is not None
    assert results.final_analytical.shape == (10, 10)

def test_new_sw_ics():
    from src.physics.equations import ShallowWaterEquation
    from src.numerics.operators import gradient
    from src.numerics.integrators import rk4
    from src.mesh.boundaries import reflect
    from src.physics.init_conditions import LocalizedDamBreakIC, CircularDamBreakIC, ReservoirIC
    
    # 1. Localized Dam Break
    config1 = SimulationConfig(
        shape=(10, 10), spacing=(0.1, 0.1), dt=0.001, final_time=0.002, steps_per_frame=1,
        equation=ShallowWaterEquation(), operator=gradient, boundary=reflect, integrator=rk4,
        initial_condition=LocalizedDamBreakIC()
    )
    sim1 = Simulation(config1)
    results1 = sim1.run()
    assert results1.final_state.data.shape == (3, 10, 10)

    # 2. Circular Dam Break
    config2 = SimulationConfig(
        shape=(10, 10), spacing=(0.1, 0.1), dt=0.001, final_time=0.002, steps_per_frame=1,
        equation=ShallowWaterEquation(), operator=gradient, boundary=reflect, integrator=rk4,
        initial_condition=CircularDamBreakIC()
    )
    sim2 = Simulation(config2)
    results2 = sim2.run()
    assert results2.final_state.data.shape == (3, 10, 10)

    # 3. Reservoir
    config3 = SimulationConfig(
        shape=(10, 10), spacing=(0.1, 0.1), dt=0.001, final_time=0.002, steps_per_frame=1,
        equation=ShallowWaterEquation(), operator=gradient, boundary=reflect, integrator=rk4,
        initial_condition=ReservoirIC()
    )
    sim3 = Simulation(config3)
    results3 = sim3.run()
    assert results3.final_state.data.shape == (3, 10, 10)

if __name__ == "__main__":
    test_simulation_runs()
    test_2d_shallow_water_rk4()
    test_2d_wave_validation()
    test_new_sw_ics()
    print("All integration tests passed!")
