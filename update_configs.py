import os

replacements = {
    "init_conditions.advec_gauss_2d": "init_conditions.GaussianIC(sigma=2.0)",
    "init_conditions.diff_gauss_2d": "init_conditions.GaussianIC(sigma=2.0)",
    "init_conditions.wave_gauss_2d": "init_conditions.GaussianIC(sigma=2.0, num_fields=2)",
    "init_conditions.advec_gauss": "init_conditions.GaussianIC(sigma=10.0, center_ratio=0.5)",
    "init_conditions.burgers_traveling_shock": "init_conditions.BurgersTravelingShockIC(nu=0.1)",
    "init_conditions.burgers_stationary_shock": "init_conditions.BurgersStationaryShockIC(nu=0.1, U=1.0)",
    "init_conditions.burgers_traveling_smooth": "init_conditions.BurgersTravelingShockIC(nu=2.0)",
    "init_conditions.diff_gauss": "init_conditions.GaussianIC(sigma=2.0, use_L_for_center=True)",
    "init_conditions.shallow_linear_gauss": "init_conditions.ShallowGaussianIC(sigma=20.0, amplitude=1e-6, ambient_depth=1.0, center_ratio=0.5, use_L_for_center=True)",
    "init_conditions.shallow_dam": "init_conditions.ShallowDamIC()",
    "init_conditions.wave_gauss": "init_conditions.GaussianIC(sigma=2.0, num_fields=2, use_L_for_center=True)",
}

for root, _, files in os.walk('configs'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            for k, v in replacements.items():
                content = content.replace(k, v)
            with open(path, 'w') as f:
                f.write(content)
print("Configs updated successfully!")
