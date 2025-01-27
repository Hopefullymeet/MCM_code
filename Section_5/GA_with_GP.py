import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deap import base, creator, tools, algorithms
from matplotlib.colors import Normalize, LogNorm

alpha0, alpha1, alpha2, alpha3 = 0.5, 0.01, 0.02, 0.03
gamma0, gamma1, gamma2 = 0.1, 0.05, 0.02
beta0, beta1, beta2 = 100, 1.2, 0.8
theta1, theta2, theta3, theta4 = 0.5, 0.3, 0.2, 0.1
omega0, omega1, omega2 = 20, 0.1, 0.05
psi0, psi1, psi2 = 0.05, 0.02, 0.01

TourNum_min, TourNum_max = 800000, 1300000
TaxRate_min, TaxRate_max = 0.05, 0.10
I_infr_base = 50
TourNum_ref = 150000

E_limit = 2.0
G_limit = 0.5
R_target = 5e5
S_target = 0.8

population_size = 300
n_generations = 50
CX_PROB = 0.7
MUT_PROB = 0.2

file_path = 'data.csv'
try:
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
except FileNotFoundError:
    print(f"文件 {file_path} 未找到。请确保文件存在于指定路径。")
    data_cleaned = pd.DataFrame(columns=["TourNum", "I_infr", "Satisfaction", "Income", "Unemployment"])

def emissions(tour_num, i_env, climate_idx=1.0):
    return alpha0 + alpha1 * tour_num - alpha2 * i_env + alpha3 * climate_idx

def glacier_melting(E_val, climate_idx=1.0):
    return gamma0 + gamma1 * E_val + gamma2 * climate_idx

def revenue(tour_num, tax_rate):
    return beta0 * (tour_num ** beta1) * (tax_rate ** beta2)

def income(i_infr, tour_num):
    val = omega0 + omega1 * i_infr + omega2 * (tour_num / TourNum_ref)
    return max(val, 1e-6)

def unemployment(i_infr, tour_num):
    return psi0 - psi1 * i_infr + psi2 * (tour_num / TourNum_ref)

def satisfaction(i_infr, tour_num, inc, unemp):
    if inc <= 0:
        return -1e9
    return (theta1 * (i_infr / I_infr_base)
            - theta2 * (tour_num / TourNum_ref)
            + theta3 * np.log(inc)
            - theta4 * unemp)

def gp_objective(individual, weights):
    wE, wG, wR, wS = weights
    frac_env, tour_num, tax_rate = individual

    R_val = revenue(tour_num, tax_rate)
    if R_val < 0:
        return (1e12,)

    i_env = frac_env * R_val
    i_infr = (1 - frac_env) * R_val

    E_val = emissions(tour_num, i_env)
    G_val = glacier_melting(E_val)
    inc = income(i_infr, tour_num)
    unemp = unemployment(i_infr, tour_num)
    S_val = satisfaction(i_infr, tour_num, inc, unemp)

    dE = max(0.0, E_val - E_limit)
    dG = max(0.0, G_val - G_limit)
    dR = max(0.0, R_target - R_val)
    dS = max(0.0, S_target - S_val)

    Z = wE * dE + wG * dG + wR * dR + wS * dS
    return (Z,)

if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def init_frac_env():
    return np.random.uniform(0.2, 0.8)

def init_tour_num():
    return np.random.uniform(TourNum_min, TourNum_max)

def init_tax_rate():
    return np.random.uniform(TaxRate_min, TaxRate_max)

toolbox.register("attr_frac_env", init_frac_env)
toolbox.register("attr_tour_num", init_tour_num)
toolbox.register("attr_tax_rate", init_tax_rate)

toolbox.register("individual",
                 tools.initCycle,
                 creator.Individual,
                 (toolbox.attr_frac_env, toolbox.attr_tour_num, toolbox.attr_tax_rate),
                 n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def cx_bounds(minvals, maxvals):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                for i in range(len(child)):
                    child[i] = min(max(child[i], minvals[i]), maxvals[i])
            return offspring
        return wrapper
    return decorator

def mut_bounds(minvals, maxvals):
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:
                for i in range(len(child)):
                    child[i] = min(max(child[i], minvals[i]), maxvals[i])
            return offspring
        return wrapper
    return decorator

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.05, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

min_bounds = [0.2, TourNum_min, TaxRate_min]
max_bounds = [0.8, TourNum_max, TaxRate_max]

toolbox.decorate("mate", cx_bounds(min_bounds, max_bounds))
toolbox.decorate("mutate", mut_bounds(min_bounds, maxvals=max_bounds))

def run_ga_scenario(weights, pop_size=300, ngen=50):
    def eval_func(ind):
        return gp_objective(ind, weights)

    toolbox.register("evaluate", eval_func)

    population = toolbox.population(n=pop_size)
    final_pop, log = algorithms.eaSimple(population, toolbox,
                                        cxpb=CX_PROB, mutpb=MUT_PROB,
                                        ngen=ngen, stats=None, halloffame=None, verbose=False)

    best_ind = tools.selBest(final_pop, k=1)[0]
    best_val = eval_func(best_ind)[0]
    return best_ind, best_val, final_pop

if __name__ == "__main__":
    wE_env, wG_env, wR_env, wS_env = 50.0, 50.0, 1.0, 1.0
    best_ind_env, best_val_env, pop_env = run_ga_scenario(
        weights=(wE_env, wG_env, wR_env, wS_env),
        pop_size=population_size,
        ngen=n_generations
    )

    wE_eco, wG_eco, wR_eco, wS_eco = 1.0, 1.0, 100.0, 1.0
    best_ind_eco, best_val_eco, pop_eco = run_ga_scenario(
        weights=(wE_eco, wG_eco, wR_eco, wS_eco),
        pop_size=population_size,
        ngen=n_generations
    )

    wE_soc, wG_soc, wR_soc, wS_soc = 1.0, 1.0, 1.0, 100.0
    best_ind_soc, best_val_soc, pop_soc = run_ga_scenario(
        weights=(wE_soc, wG_soc, wR_soc, wS_soc),
        pop_size=population_size,
        ngen=n_generations
    )

    def decode_and_print(label, best_ind, best_val):
        frac_env, tour_opt, tax_opt = best_ind
        R_opt = revenue(tour_opt, tax_opt)
        i_env_opt = frac_env * R_opt
        i_infr_opt = (1 - frac_env) * R_opt
        E_opt = emissions(tour_opt, i_env_opt)
        G_opt = glacier_melting(E_opt)
        I_opt = income(i_infr_opt, tour_opt)
        U_opt = unemployment(i_infr_opt, tour_opt)
        S_val = satisfaction(i_infr_opt, tour_opt, I_opt, U_opt)

        dE_opt = max(0.0, E_opt - E_limit)
        dG_opt = max(0.0, G_opt - G_limit)
        dR_opt = max(0.0, R_target - R_opt)
        dS_opt = max(0.0, S_target - S_val)

        print("=============================================================")
        print(f"Scenario: {label}")
        print(f"Best Individual = [frac_env={frac_env:.4f}, TourNum={tour_opt:.2f}, TaxRate={tax_opt:.4f}]")
        print(f"Best Objective Value (Sum of Weighted Deviations) = {best_val:.6f}")
        print(f"Revenue = {R_opt:.4f} (target={R_target}, dR={dR_opt:.4f})")
        print(f"Satisfaction = {S_val:.4f} (target={S_target}, dS={dS_opt:.4f})")
        print(f"Emissions = {E_opt:.4f} (limit={E_limit}, dE={dE_opt:.4f})")
        print(f"Glacier = {G_opt:.4f} (limit={G_limit}, dG={dG_opt:.4f})")
        print(f"Income = {I_opt:.4f}, Unemployment = {U_opt:.4f}")

    decode_and_print("Environment-First", best_ind_env, best_val_env)
    decode_and_print("Economy-First", best_ind_eco, best_val_eco)
    decode_and_print("Social-Priority", best_ind_soc, best_val_soc)

    def plot_3d_population(population, scenario_label, best_ind, weights):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection='3d')

        frac_vals = []
        t_vals = []
        tax_vals = []
        obj_vals = []

        for ind in population:
            frac_env, tour_num, tax_rate = ind
            Z = gp_objective(ind, weights)[0]
            frac_vals.append(frac_env)
            t_vals.append(tour_num)
            tax_vals.append(tax_rate)
            obj_vals.append(Z)

        print(f"Scenario: {scenario_label}")
        print(f"Objective Value - min: {min(obj_vals)}, max: {max(obj_vals)}, mean: {np.mean(obj_vals)}, std: {np.std(obj_vals)}")

        norm = Normalize(vmin=min(obj_vals), vmax=max(obj_vals))

        cmap_options = ['viridis', 'plasma', 'inferno', 'cividis']

        for cmap in cmap_options:
            plt.figure(figsize=(9, 6))
            scatter = ax.scatter(frac_vals, t_vals, tax_vals, c=obj_vals, cmap=cmap, norm=norm, alpha=0.6, label='Population')
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
            cbar.set_label('Objective Value')
            ax.set_xlabel('I_env (frac_env * Revenue)')
            ax.set_ylabel('TourNum')
            ax.set_zlabel('TaxRate')
            ax.set_title(f'Final Population: {scenario_label} - {cmap}')
            ax.legend()

            best_frac_env, best_tour_num, best_tax_rate = best_ind
            best_obj_val = gp_objective(best_ind, weights)[0]
            ax.scatter(best_frac_env, best_tour_num, best_tax_rate,
                       color='red', s=100, label='Best Individual')

            filename = f'{scenario_label.lower().replace(" ","_")}_3d_scatter_{cmap}.png'
            plt.savefig(filename, dpi=120)
            print(f"Saved 3D scatter plot for {scenario_label} with {cmap} as: {filename}")
            plt.show()

    plot_3d_population(pop_env, "Environment-First", best_ind_env, weights=(wE_env, wG_env, wR_env, wS_env))
    plot_3d_population(pop_eco, "Economy-First", best_ind_eco, weights=(wE_eco, wG_eco, wR_eco, wS_eco))
    plot_3d_population(pop_soc, "Social-Priority", best_ind_soc, weights=(wE_soc, wG_soc, wR_soc, wS_soc))
