import os
import re

for root, _, files in os.walk('configs'):
    for file in files:
        if file.endswith('.py'):
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
            
            # Find the coefficients_list
            match = re.search(r'coefficients\s*=\s*\[(.*?)\]', content)
            if not match:
                continue
                
            coeff_val = match.group(1).strip()
            
            # Remove coefficients list
            content = re.sub(r'coefficients\s*=\s*\[.*?\]\n?', '', content)
            
            # Replace equations
            content = content.replace('equations.advection', f'equations.AdvectionEquation(velocity={coeff_val})')
            content = content.replace('equations.wave', f'equations.WaveEquation(wave_speed={coeff_val})')
            content = content.replace('equations.diffusion', f'equations.DiffusionEquation(diffusivity={coeff_val})')
            content = content.replace('equations.shallow_water', 'equations.ShallowWaterEquation()')
            content = content.replace('equations.burgers', f'equations.BurgersEquation(viscosity={coeff_val})')
            
            with open(path, 'w') as f:
                f.write(content)

print("Config equations refactored successfully!")
