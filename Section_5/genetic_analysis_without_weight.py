import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools, algorithms

alpha0, alpha1, alpha2, alpha3 = 0.5, 0.01, 0.02, 0.03
gamma0, gamma1, gamma2 = 0.1, 0.05, 0.02
beta0, beta1, beta2 = 100, 1.2, 0.8
theta1, theta2, theta3, theta4 = 0.5, 0.3, 0.2, 0.1
omega0, omega1, omega2 = 20, 0.1, 0.05
psi0, psi1, psi2 = 0.05, 0.02, 0.01

TourNum_min, TourNum_max = 800000, 1300000
Iinfr_base = 50
TourNum_ref = 150000

R_target = 5e5
S_target = 0.8

population_size = 300
generations = 50

file_path = 'data.csv'
data = pd.read_csv(file_path)

columns = [
    "Visitor_Number",
    "Infrastructure_Investment_%",
    "Resident_Satisfaction",
    "Total_Income_million",
    "Unemployment_Rate",
]
data_cleaned = data[columns].dropna()
data_cleaned.columns = ["TourNum", "I_infr", "Satisfaction", "Income", "Unemployment"]

def emissions(TourNum, Ienv, ClimateIndex=1):
    return alpha0 + alpha1 * TourNum - alpha2 * Ienv + alpha3 * ClimateIndex

def glacier_melting(E, ClimateIndex=1):
    return gamma0 + gamma1 * E + gamma2 * ClimateIndex

def revenue(TourNum, TaxRate):
    return beta0 * (TourNum ** beta1) * (TaxRate ** beta2)

def satisfaction(Iinfr, TourNum, Income, Unemployment):
    if Income <= 0:
        return -1e6
    return (theta1 * (Iinfr / Iinfr_base)
            - theta2 * (TourNum / TourNum_ref)
            + theta3 * np.log(Income)
            - theta4 * Unemployment)

def income(Iinfr, TourNum):
    if Iinfr < 0:
        return 1e-6
    val = omega0 + omega1 * Iinfr + omega2 * (TourNum / TourNum_ref)
    return max(val, 1e-6)

def unemployment(Iinfr, TourNum):
    return psi0 - psi1 * Iinfr + psi2 * (TourNum / TourNum_ref)

def objective(individual):
    a, TourNum, TaxRate = individual
    rev = revenue(TourNum, TaxRate)
    Ienv = a * rev
    Iinfr = (1 - a) * rev
    
    E = emissions(TourNum, Ienv)
    G = glacier_melting(E)
    R = rev
    I = income(Iinfr, TourNum)
    U = unemployment(Iinfr, TourNum)
    S = satisfaction(Iinfr, TourNum, I, U)
    
    raw_obj = 0.4 * S + 0.3 * R - 0.2 * E - 0.1 * G
    
    penalty = 0.0
    if R < R_target:
        penalty += (R_target - R) * 0.001
    if S < S_target:
        penalty += (S_target - S) * 500
    
    return -(raw_obj - penalty),

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

toolbox.register("attr_a", np.random.uniform, 0, 1)
toolbox.register("attr_TourNum", np.random.uniform, TourNum_min, TourNum_max)
toolbox.register("attr_TaxRate", np.random.uniform, 0.05, 0.10)

toolbox.register("individual",
                 tools.initCycle,
                 creator.Individual,
                 (toolbox.attr_a, toolbox.attr_TourNum, toolbox.attr_TaxRate))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", objective)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def checkBoundsA_TourNum_TaxRate(min_a, max_a, min_tn, max_tn, min_tax, max_tax):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                if child[0] < min_a: child[0] = min_a
                if child[0] > max_a: child[0] = max_a
                if child[1] < min_tn: child[1] = min_tn
                if child[1] > max_tn: child[1] = max_tn
                if child[2] < min_tax: child[2] = min_tax
                if child[2] > max_tax: child[2] = max_tax
            return offspring
        return wrapper
    return decorator

toolbox.decorate("mate", checkBoundsA_TourNum_TaxRate(0, 1, TourNum_min, TourNum_max, 0.05, 0.10))
toolbox.decorate("mutate", checkBoundsA_TourNum_TaxRate(0, 1, TourNum_min, TourNum_max, 0.05, 0.10))

if __name__ == "__main__":
    population = toolbox.population(n=population_size)
    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=generations,
        stats=None,
        halloffame=None,
        verbose=True
    )

    best_individual = tools.selBest(population, k=1)[0]
    a_optimal, TourNum_optimal, TaxRate_optimal = best_individual
    optimal_objective = -objective(best_individual)[0]

    print("\nOptimal Decision Variables:")
    print(f"Investment Proportion (a): {a_optimal:.4f}")
    print(f"Tourists (TourNum): {TourNum_optimal:.2f}")
    print(f"Tax Rate: {TaxRate_optimal:.4f}")
    print(f"Optimal Objective Value: {optimal_objective:.4f}")

    rev = revenue(TourNum_optimal, TaxRate_optimal)
    Ienv = a_optimal * rev
    Iinfr = (1 - a_optimal) * rev
    E_opt = emissions(TourNum_optimal, Ienv)
    G_opt = glacier_melting(E_opt)
    R_opt = rev
    I_opt = income(Iinfr, TourNum_optimal)
    U_opt = unemployment(Iinfr, TourNum_optimal)
    S_opt = satisfaction(Iinfr, TourNum_optimal, I_opt, U_opt)

    print(f"Revenue (R) = {R_opt:.2f} (target={R_target})")
    print(f"Satisfaction (S) = {S_opt:.4f} (target={S_target})")
    print(f"Emissions (E) = {E_opt:.4f}")
    print(f"Glacier Melting (G) = {G_opt:.4f}")
    print(f"Income (I) = {I_opt:.4f}")
    print(f"Unemployment (U) = {U_opt:.4f}")

    a_vals = []
    TourNum_vals = []
    TaxRate_vals = []
    obj_vals = []

    for ind in population:
        a, tn, tr = ind
        val = -objective(ind)[0]
        a_vals.append(a)
        TourNum_vals.append(tn)
        TaxRate_vals.append(tr)
        obj_vals.append(val)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(a_vals, TourNum_vals, TaxRate_vals,
                         c=obj_vals, cmap='viridis', marker='o', alpha=0.6, label='Solutions')
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Objective Value')

    ax.scatter(a_optimal, TourNum_optimal, TaxRate_optimal,
               color='red', s=150, label='Optimal Point')

    ax.set_xlabel('I_infr', fontsize=12)
    ax.set_ylabel('TourNum', fontsize=12)
    ax.set_zlabel('Tax Rate', fontsize=12)
    ax.set_title('Optimization Results: I_infr, TourNum, TaxRate vs Objective Value', fontsize=16)
    ax.legend()
    plt.show()

    plot_path = 'optimization_results_3d_scatter.png'
    fig.savefig(plot_path)
    print(f"Optimization 3D scatter plot saved at: {plot_path}")

    fixed_tax_rates = [0.05, 0.075, 0.10]

    fig_surface = plt.figure(figsize=(18, 6))
    for i, tr in enumerate(fixed_tax_rates):
        ax_surface = fig_surface.add_subplot(1, len(fixed_tax_rates), i+1, projection='3d')
        
        a_range = np.linspace(0, 1, 20)
        TourNum_range = np.linspace(TourNum_min, TourNum_max, 20)
        A, TourNum_grid = np.meshgrid(a_range, TourNum_range)
        TaxRate_grid = tr * np.ones_like(A)
        
        Z = []
        for a_val, tn_val, tr_val in zip(A.flatten(), TourNum_grid.flatten(), TaxRate_grid.flatten()):
            val = -objective([a_val, tn_val, tr_val])[0]
            Z.append(val)
        Z = np.array(Z).reshape(A.shape)
        
        surf = ax_surface.plot_surface(A, TourNum_grid, Z, cmap='viridis', alpha=0.7)

        if abs(tr - TaxRate_optimal) < 0.005:
            ax_surface.scatter(a_optimal, TourNum_optimal, -objective([a_optimal, TourNum_optimal, tr])[0],
                               color='red', s=100, label='Optimal Point')
        
        ax_surface.set_xlabel('I_infr')
        ax_surface.set_ylabel('TourNum')
        ax_surface.set_zlabel('Objective Value')
        ax_surface.set_title(f'Tax Rate = {100*tr:.1f}%')
        ax_surface.legend()

    plt.suptitle('Objective Function Surface with Fixed Tax Rates (5%, 7.5%, 10%)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    surface_plot_path = 'objective_surface_fixed_tax_rates.png'
    fig_surface.savefig(surface_plot_path)
    print(f"Objective surface plots saved at: {surface_plot_path}")
