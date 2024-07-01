import numpy as np
from scipy.optimize import minimize
import random
import warnings

PARAMS_NUM = 4
warnings.filterwarnings("ignore", category=RuntimeWarning)


def our_law_transform(log_x, A, B, beta, E):
    """
    Transform the input x to the prediction y using our law based on four parameters A, B, beta, and E.
    Note that input x is in log scale.
    """
    expnum = beta * log_x + B
    return np.log((A-abs(E)) /(1 + np.exp(expnum)) + abs(E))

def vanilla_law_transform(log_x, alpha, B, beta, E):
    """
    Transform the input x to the prediction y using vanilla law based on four parameters alpha, B, beta, and E.
    Note that input x is in log scale.
    """
    return alpha * np.log(1 / (B * (np.exp(log_x) ** beta)) + abs(E))

def fit_vanilla_law(log_x, log_y):
    """
    Fit a vanilla law to the data.
    Note that both x and y are in log scale.
    """
    def huber_loss_func(true, pred, delta=(0.001)):
        error = true - pred
        is_small_error = np.abs(error) < delta
        squared_loss = np.square(error) / 2
        linear_loss = delta * (np.abs(error) - delta / 2)
        return np.where(is_small_error, squared_loss, linear_loss)
    
    def objective(params):
        log_y_pred = vanilla_law_transform(log_x, *params)
        return np.sum(huber_loss_func(log_y, log_y_pred))

    best_loss = float('inf')
    best_params = None

    # repeat for 10 times
    for _ in range(10): 
        initial_guess = [random.random() for _ in range(PARAMS_NUM)]
        result = minimize(objective, initial_guess)
        if result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x

    return best_params, best_loss

def fit_our_law(log_x, log_y):
    """
    Fit our law to the data.
    Note that both x and y are in log scale.
    """
    def surrogate_optimize(params):
        params = np.concatenate([init_params, np.ones_like(log_x)])

        def first_loss(surr_params):
            Z = surr_params[PARAMS_NUM:]
            Z = Z**2 # Z = exp(beta*x + B) > 0
            A, B, beta, E = surr_params[:PARAMS_NUM]

            # MSE loss 
            Z = Z
            y_loss = np.square(np.log(A + np.abs(E) * Z) - np.log(1 + Z) - log_y).mean() if Z.shape[0] > 0 else 0
            x_loss = np.square(np.log(Z) - beta * log_x - B).mean() if Z.shape[0] > 0 else 0

            return x_loss + y_loss # + 0.001 * np.abs(E)

        result = minimize(first_loss, params)
        params = result.x
        
        # Fix beta and D and re-tune A and E
        A, B, beta, E = params[:PARAMS_NUM]
        def second_loss(intercepts):
            _A, _E = intercepts
            log_y_pred = our_law_transform(log_x, _A, B, beta, _E)

            y_loss = np.square(log_y_pred - log_y).mean()

            return y_loss

        result = minimize(second_loss, [A, E])
        A, E = result.x
        params = [A, B, beta, E]
        return params, result.fun

    best_loss = float('inf')
    best_params = None
    u, l =np.max(log_y), np.min(log_y)

    # repeat for 3 times
    for i in range(3):
        # initialization
        A, E = np.exp(u), np.exp((u + l) / 2)
        init_params = np.array([A, 0.0, 1.0, E])
        
        params, lss = surrogate_optimize(init_params)

        if lss < best_loss and params[0] > params[3]:
            best_loss = lss
            best_params = params

    return best_params, best_loss