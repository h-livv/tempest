import os
import re

replacements = {
    r'from src\.grid': r'from src.mesh.grid',
    r'from src\.fields': r'from src.mesh.fields',
    r'from src\.boundaries': r'from src.mesh.boundaries',
    r'from src\.operators': r'from src.numerics.operators',
    r'from src\.integrators': r'from src.numerics.integrators',
    r'from src\.direct_solvers': r'from src.numerics.direct_solvers',
    r'from src\.equations': r'from src.physics.equations',
    r'from src\.init_conditions': r'from src.physics.init_conditions',
    r'from src\.diagnostics\.validation': r'from src.validation.validation',
    r'from src\.diagnostics import .*validation': r'from src.validation import validation\nfrom src.diagnostics import stability',
    r'from visualizations\.visualization': r'from src.visualization.visualization',
}

module_mapping = {
    'boundaries': 'src.mesh',
    'operators': 'src.numerics',
    'integrators': 'src.numerics',
    'direct_solvers': 'src.numerics',
    'equations': 'src.physics',
    'init_conditions': 'src.physics',
    'grid': 'src.mesh',
    'fields': 'src.mesh'
}

for root, _, files in os.walk('z:\\home\\harliv\\projects\\tempest'):
    if '.git' in root or '.venv' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py') or file.endswith('.md'):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            new_content = content
            
            # Simple replacements
            for k, v in replacements.items():
                if k == r'from src\.diagnostics import .*validation':
                    new_content = re.sub(r'from src\.diagnostics import (.*)validation(.*)', r'from src.validation import validation\nfrom src.diagnostics import \1\2', new_content)
                    new_content = new_content.replace('from src.diagnostics import stability
', 'from src.diagnostics import stability\n')
                    new_content = new_content.replace('from src.diagnostics import \n', '')
                    new_content = new_content.replace('from src.validation import validation
from src.diagnostics import stability\nvalidation', 'from src.diagnostics import stability\nfrom src.validation import ')
                else:
                    new_content = re.sub(k, v, new_content)
                
            # Handle "from src import X, Y, Z"
            def replace_src_import(match):
                modules = [m.strip() for m in match.group(1).split(',')]
                groups = {}
                for m in modules:
                    if m in module_mapping:
                        pkg = module_mapping[m]
                        groups.setdefault(pkg, []).append(m)
                    else:
                        groups.setdefault('src', []).append(m)
                
                res = []
                for pkg, mods in groups.items():
                    res.append(f"from {pkg} import {', '.join(mods)}")
                return '\n'.join(res)

            new_content = re.sub(r'from src import ([\w\s,]+)', replace_src_import, new_content)

            # Special cases
            new_content = new_content.replace('from src.validation import validation
from src.diagnostics import stability
from src.validation import validation', 'from src.diagnostics import stability\nfrom src.validation import ')

            if new_content != content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Updated {path}")
