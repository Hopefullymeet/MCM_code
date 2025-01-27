import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

data_path = "1.xlsx"
sheet_data = pd.read_excel(data_path, sheet_name="Sheet1")

print("Columns in the data:", sheet_data.columns)

TourNum_min = sheet_data["Visitor_Number"].min()
TourNum_max = sheet_data["Visitor_Number"].max()

TaxRate_min, TaxRate_max = 0.05, 0.10

E_limit = 2.0
G_limit = 0.5
R_target = sheet_data["RevenueM&euro;"].mean()
S_target = sheet_data["Resident_Satisfaction"].mean()

population_size = 200
n_generations = 30
CX_PROB = 0.7
MUT_PROB = 0.2

alpha0, alpha1, alpha2, alpha3 = 0.5, 0.01, 0.02, 0.03
gamma0, gamma1, gamma2 = 0.1, 0.05, 0.02
beta0, beta1, beta2 = 100, 1.2, 0.8

theta1, theta2, theta3, theta4 = 0.5, 0.3, 0.2, 0.1
omega0, omega1, omega2 = 20, 0.1, 0.05
psi0, psi1, psi2 = 0.05, 0.02, 0.01

I_infr_base = 50
TourNum_ref = 150000

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

def gp_objective(ind, weights, E_limit, G_limit, R_target, S_target):
    wE, wG, wR, wS = weights
    frac_env, tour_num, tax_rate = ind

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

min_bounds = [0.2, TourNum_min, TaxRate_min]
max_bounds = [0.8, TourNum_max, TaxRate_max]

def init_frac_env():
    return np.random.uniform(0.2, 0.8)

def init_tour_num():
    return np.random.uniform(TourNum_min, TourNum_max)

def init_tax_rate():
    return np.random.uniform(TaxRate_min, TaxRate_max)

toolbox.register("attr_frac_env", init_frac_env)
toolbox.register("attr_tour_num", init_tour_num)
toolbox.register("attr_tax_rate", init_tax_rate)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_frac_env, toolbox.attr_tour_num, toolbox.attr_tax_rate),
                 n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.05, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

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

toolbox.decorate("mate", cx_bounds(min_bounds, max_bounds))
toolbox.decorate("mutate", mut_bounds(min_bounds, max_bounds))

def run_ga_scenario(weights, E_limit, G_limit, R_target, S_target,
                    pop_size=200, ngen=30, cxpb=0.7, mutpb=0.2):
    def eval_func(ind):
        return gp_objective(ind, weights, E_limit, G_limit, R_target, S_target)
    toolbox.register("evaluate", eval_func)

    population = toolbox.population(n=pop_size)
    final_pop, log = algorithms.eaSimple(population, toolbox,
                                         cxpb=cxpb, mutpb=mutpb,
                                         ngen=ngen, stats=None,
                                         halloffame=None, verbose=False)

    best_ind = tools.selBest(final_pop, k=1)[0]
    best_val = eval_func(best_ind)[0]
    return best_ind, best_val, final_pop

def decode_solution(ind, E_limit, G_limit, R_target, S_target):
    frac_env, tour_num, tax_opt = ind
    R_opt = revenue(tour_num, tax_opt)
    i_env_opt = frac_env * R_opt
    i_infr_opt = (1 - frac_env) * R_opt
    E_opt = emissions(tour_num, i_env_opt)
    G_opt = glacier_melting(E_opt)
    inc = income(i_infr_opt, tour_num)
    unemp = unemployment(i_infr_opt, tour_num)
    S_val = satisfaction(i_infr_opt, tour_num, inc, unemp)

    dE = max(0, E_opt - E_limit)
    dG = max(0, G_opt - G_limit)
    dR = max(0, R_target - R_opt)
    dS = max(0, S_target - S_val)

    overE = max(0, E_limit - E_opt)
    overG = max(0, G_limit - G_opt)
    overR = max(0, R_opt - R_target)
    overS = max(0, S_val - S_target)

    return {
        'frac_env': frac_env, 'TourNum': tour_num, 'TaxRate': tax_opt,
        'E': E_opt, 'G': G_opt, 'R': R_opt, 'S': S_val,
        'dE': dE, 'dG': dG, 'dR': dR, 'dS': dS,
        'overE': overE, 'overG': overG, 'overR': overR, 'overS': overS
    }

if __name__ == "__main__":
    wE_env, wG_env, wR_env, wS_env = 50.0, 50.0, 1.0, 1.0
    wE_eco, wG_eco, wR_eco, wS_eco = 1.0, 1.0, 100.0, 1.0
    wE_soc, wG_soc, wR_soc, wS_soc = 1.0, 1.0, 1.0, 100.0

    bestE, valE, popE = run_ga_scenario((wE_env, wG_env, wR_env, wS_env),
                                        E_limit, G_limit, R_target, S_target,
                                        pop_size=population_size, ngen=n_generations)
    sol_env = decode_solution(bestE, E_limit, G_limit, R_target, S_target)

    bestR, valR, popR = run_ga_scenario((wE_eco, wG_eco, wR_eco, wS_eco),
                                        E_limit, G_limit, R_target, S_target,
                                        pop_size=population_size, ngen=n_generations)
    sol_eco = decode_solution(bestR, E_limit, G_limit, R_target, S_target)

    bestS, valS, popS = run_ga_scenario((wE_soc, wG_soc, wR_soc, wS_soc),
                                        E_limit, G_limit, R_target, S_target,
                                        pop_size=population_size, ngen=n_generations)
    sol_soc = decode_solution(bestS, E_limit, G_limit, R_target, S_target)

    print("Environment-First:", sol_env)
    print("Economy-First:", sol_eco)
    print("Social-Priority:", sol_soc)

    scenarios = ["Env-First", "Eco-First", "Soc-First"]
    devE = [sol_env['dE'], sol_eco['dE'], sol_soc['dE']]
    devG = [sol_env['dG'], sol_eco['dG'], sol_soc['dG']]
    devR = [sol_env['dR'], sol_eco['dR'], sol_soc['dR']]
    devS = [sol_env['dS'], sol_eco['dS'], sol_soc['dS']]

    x = np.arange(len(scenarios))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x - 1.5*width, devE, width, label='dE')
    ax.bar(x - 0.5*width, devG, width, label='dG')
    ax.bar(x + 0.5*width, devR, width, label='dR')
    ax.bar(x + 1.5*width, devS, width, label='dS')
    ax.set_title('Shortfall (Positive Deviation) under different priorities')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Shortfall from Targets')
    ax.legend()
    plt.tight_layout()
    plt.show()

    overE_vals = [sol_env['overE'], sol_eco['overE'], sol_soc['overE']]
    overG_vals = [sol_env['overG'], sol_eco['overG'], sol_soc['overG']]
    overR_vals = [sol_env['overR'], sol_eco['overR'], sol_soc['overR']]
    overS_vals = [sol_env['overS'], sol_eco['overS'], sol_soc['overS']]

    fig2, ax2 = plt.subplots(figsize=(8,5))
    neg_overE = [-v for v in overE_vals]
    neg_overG = [-v for v in overG_vals]
    neg_overR = [-v for v in overR_vals]
    neg_overS = [-v for v in overS_vals]

    ax2.bar(x - 1.5*width, neg_overE, width, label='Over-E', color='blue')
    ax2.bar(x - 0.5*width, neg_overG, width, label='Over-G', color='orange')
    ax2.bar(x + 0.5*width, neg_overR, width, label='Over-R', color='green')
    ax2.bar(x + 1.5*width, neg_overS, width, label='Over-S', color='red')
    ax2.set_title('Overachieve Visualization (Negative = better than target)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.set_ylabel('Negative = how much better than target')
    ax2.legend()
    plt.tight_layout()
    plt.show()
