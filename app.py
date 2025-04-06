import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import streamlit as st

# Portfolio statistics
def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

def portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)

# Constraint functions
def get_constraints(target_return, expected_returns):
    return [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: portfolio_return(w, expected_returns) - target_return}
    ]

# Bounds (e.g., no short-selling)
def get_bounds(n_assets):
    return tuple((0, 1) for _ in range(n_assets))

# Solver function
def solve_min_variance_portfolio(expected_returns, cov_matrix, target_return):
    n_assets = len(expected_returns)
    init_guess = np.ones(n_assets) / n_assets
    constraints = get_constraints(target_return, expected_returns)
    bounds = get_bounds(n_assets)

    result = minimize(portfolio_variance,
                      init_guess,
                      args=(cov_matrix,),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    return result

# Generate sample data
def generate_sample_data():
    returns = np.array([0.12, 0.10, 0.07])
    cov_matrix = np.array([
        [0.0064, 0.0008, 0.0011],
        [0.0008, 0.0025, 0.0014],
        [0.0011, 0.0014, 0.0036]
    ])
    return returns, cov_matrix

# Plot Minimum Variance Frontier
def plot_min_variance_frontier(expected_returns, cov_matrix, num_points=100):
    target_returns = np.linspace(min(expected_returns), max(expected_returns), num_points)
    risks = []
    weights_list = []

    for target_return in target_returns:
        result = solve_min_variance_portfolio(expected_returns, cov_matrix, target_return)
        if result.success:
            risk = np.sqrt(portfolio_variance(result.x, cov_matrix))
            risks.append(risk)
            weights_list.append(result.x)
        else:
            risks.append(np.nan)
            weights_list.append([np.nan]*len(expected_returns))

    plt.figure(figsize=(10, 6))
    plt.plot(risks, target_returns, marker='o', linestyle='-', label='Minimum Variance Frontier')
    plt.xlabel('Portfolio Risk (Standard Deviation)')
    plt.ylabel('Portfolio Return')
    plt.title('Minimum Variance Frontier')
    plt.grid(True)
    plt.legend()
    st.pyplot(plt.gcf())
    return target_returns, risks, weights_list

# Streamlit Web App
def main():
    st.title("Minimum Variance Portfolio Optimizer")

    expected_returns, cov_matrix = generate_sample_data()
    st.write("### Asset Expected Returns")
    st.write(expected_returns)
    st.write("### Covariance Matrix")
    st.write(pd.DataFrame(cov_matrix))

    st.write("---")
    target_return = st.slider("Select Target Return", float(min(expected_returns)), float(max(expected_returns)), 0.09, step=0.001)
    result = solve_min_variance_portfolio(expected_returns, cov_matrix, target_return)

    if result.success:
        st.subheader("Optimization Result")
        st.write("**Optimal Weights:**", result.x)
        st.write("**Minimum Variance:**", result.fun)
        st.write("**Portfolio Return:**", portfolio_return(result.x, expected_returns))
    else:
        st.error("Optimization failed: " + result.message)

    st.write("---")
    st.subheader("Minimum Variance Frontier")
    plot_min_variance_frontier(expected_returns, cov_matrix)

if __name__ == "__main__":
    main()
