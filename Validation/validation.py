import numpy as np

def l2_error(numerical, analytical):
    return np.sqrt(np.mean((numerical - analytical)**2))

def relative_error(numerical, analytical):
    return (np.linalg.norm(numerical - analytical)/np.linalg.norm(analytical))
    

def validation(equation, state, init_condition, N, x, t, c):
    
    if equation.__name__ == "advection":
        L = x[-1] - x[0]

        x_shifted = (x - c*t - x[0]) % L + x[0]
        analytic_state = init_condition(N, x_shifted)[0]
    
        l2 = l2_error(state, analytic_state)
        relative = relative_error(state, analytic_state)
        max_error = np.max(np.abs(state - analytic_state))
        
        #if equation.__name__ == "wave":
            
    
    return {"l2_error": l2, "relative_error": relative, "max_error": max_error, "relative": state, "analytic_state": analytic_state}


    