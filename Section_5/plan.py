import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

alpha0, alpha1, alpha2, alpha3 = 0.5, 0.01, 0.02, 0.03
gamma0, gamma1, gamma2 = 0.1, 0.05, 0.02
beta0,  beta1,  beta2 = 100, 1.2, 0.8
theta1, theta2, theta3, theta4 = 0.5, 0.3, 0.2, 0.1
omega0, omega1, omega2 = 20, 0.1, 0.05
psi0, psi1, psi2 = 0.05, 0.02, 0.01

TourNum_min, TourNum_max = 800_000, 1_300_000
TaxRate_min, TaxRate_max = 0.05, 0.10
TourNum_ref = 150_000
I_infr_base = 50
I_env_base = 50

E_limit = 2.0
G_limit = 0.5
R_target = 5e5
S_target = 0.8

def emissions(tour_num, i_env, climate_idx=1.0):
    """
    E = alpha0 + alpha1 * tour_num - alpha2 * i_env + alpha3 * climate_idx
    """
    return alpha0 + alpha1 * tour_num - alpha2 * i_env + alpha3 * climate_idx

def glacier_melting(E_val, climate_idx=1.0):
    """
    G = gamma0 + gamma1 * E_val + gamma2 * climate_idx
    """
    return gamma0 + gamma1 * E_val + gamma2 * climate_idx

def revenue(tour_num, tax_rate):
    """
    R = beta0 * (tour_num^beta1) * (tax_rate^beta2)
    """
    return beta0 * (tour_num ** beta1) * (tax_rate ** beta2)

def income(i_infr, tour_num):
    """
    I = omega0 + omega1 * i_infr + omega2 * (tour_num / TourNum_ref)
    """
    val = omega0 + omega1 * i_infr + omega2 * (tour_num / TourNum_ref)
    return max(val, 1e-6)

def unemployment(i_infr, tour_num):
    """
    U = psi0 - psi1 * i_infr + psi2 * (tour_num / TourNum_ref)
    """
    return psi0 - psi1 * i_infr + psi2 * (tour_num / TourNum_ref)

def satisfaction(i_infr, tour_num, inc, unemp):
    """
    S = theta1*(i_infr/I_infr_base) - theta2*(tour_num/TourNum_ref) + theta3*ln(inc) - theta4*unemp
    """
    if inc <= 0:
        return -1e9
    return (theta1 * (i_infr / I_infr_base)
            - theta2 * (tour_num / TourNum_ref)
            + theta3 * np.log(inc)
            - theta4 * unemp)

def tres_objective(individual):
    frac_env, tour_num, tax_rate = individual

    R_val = revenue(tour_num, tax_rate)
    if R_val < 0:
        return (1e12,)

    i_env = frac_env * R_val
    i_infr = (1 - frac_env) * R_val

    E_val = emissions(tour_num, i_env)
    G_val = glacier_melting(E_val)

    Inc = income(i_infr, tour_num)
    Unemp = unemployment(i_infr, tour_num)
    S_val = satisfaction(i_infr, tour_num, Inc, Unemp)

    raw_obj = 0.4 * S_val + 0.3 * R_val - 0.2 * E_val - 0.1 * G_val

    penalty = 0.0
    if R_val < R_target:
        penalty += (R_target - R_val) * 0.001
    if S_val < S_target:
        penalty += (S_target - S_val) * 500

    fitness_value = -(raw_obj) + penalty
    return (fitness_value,)

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

toolbox.register("evaluate", tres_objective)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def bound_check(func):
    def wrapper(*args, **kwargs):
        offspring = func(*args, **kwargs)
        for child in offspring:
            if child[0] < 0.2:
                child[0] = 0.2
            if child[0] > 0.8:
                child[0] = 0.8
            if child[1] < TourNum_min:
                child[1] = TourNum_min
            if child[1] > TourNum_max:
                child[1] = TourNum_max
            if child[2] < TaxRate_min:
                child[2] = TaxRate_min
            if child[2] > TaxRate_max:
                child[2] = TaxRate_max
        return offspring
    return wrapper

toolbox.decorate("mate", bound_check)
toolbox.decorate("mutate", bound_check)

def run_optimization(pop_size=200, ngen=30, cxpb=0.7, mutpb=0.2):
    pop = toolbox.population(n=pop_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    hof = tools.HallOfFame(1)
    logbook = tools.Logbook()

    pop, logbook = algorithms.eaSimple(pop, toolbox,
                                       cxpb=cxpb,
                                       mutpb=mutpb,
                                       ngen=ngen,
                                       stats=stats,
                                       halloffame=hof,
                                       verbose=True)
    best_ind = hof[0]

    normalized_I_envs = []
    normalized_I_infras = []
    for ind in pop:
        R_val = revenue(ind[1], ind[2])
        i_env = ind[0] * R_val
        i_infr = (1 - ind[0]) * R_val
        I_env_total = i_env
        I_infr_total = i_infr
        normalized_I_env = I_env_total / (I_env_total + I_infr_total)
        normalized_I_infr = I_infr_total / (I_env_total + I_infr_total)
        normalized_I_envs.append(normalized_I_env)
        normalized_I_infras.append(normalized_I_infr)

    return best_ind, pop, logbook, normalized_I_envs, normalized_I_infras

if __name__ == "__main__":
    best_solution, final_population, logbook, normalized_I_envs, normalized_I_infras = run_optimization(pop_size=300, ngen=50)

    print("\n==== Best Solution Found ====")
    frac_env_opt, tour_opt, tax_opt = best_solution
    print(f"Optimal frac_env = {frac_env_opt:.4f}")
    print(f"Optimal TourNum  = {tour_opt:.2f}")
    print(f"Optimal TaxRate  = {tax_opt:.4f}")

    R_opt = revenue(tour_opt, tax_opt)
    i_env_opt = frac_env_opt * R_opt
    i_infr_opt = (1 - frac_env_opt) * R_opt
    E_opt = emissions(tour_opt, i_env_opt)
    G_opt = glacier_melting(E_opt)
    Inc_opt = income(i_infr_opt, tour_opt)
    U_opt = unemployment(i_infr_opt, tour_opt)
    S_opt = satisfaction(i_infr_opt, tour_opt, Inc_opt, U_opt)

    print(f"Revenue (R)       = {R_opt:.2f}")
    print(f"I_env            = {i_env_opt:.2f}, I_infr = {i_infr_opt:.2f}")
    print(f"Emissions (E)    = {E_opt:.4f}")
    print(f"Glacier (G)      = {G_opt:.4f}")
    print(f"Income (I)       = {Inc_opt:.4f}")
    print(f"Unemployment (U) = {U_opt:.4f}")
    print(f"Satisfaction (S) = {S_opt:.4f}")

    fitnesses = [ind.fitness.values[0] for ind in final_population]

    try:
        best_index = final_population.index(best_solution)
    except ValueError:
        best_index = 0
    normalized_I_env_opt = normalized_I_envs[best_index]
    normalized_I_infr_opt = normalized_I_infras[best_index]

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(normalized_I_envs, normalized_I_infras, c='purple', alpha=0.6)
    plt.xlabel('I_env / (I_env + I_infr)')
    plt.ylabel('I_infr / (I_env + I_infr)')
    plt.title('Normalized Investment Distribution')
    plt.grid(True)
    plt.plot([0, 1], [1, 0], 'r--')

    plt.subplot(1, 2, 2)
    labels = ['I_env / Total Investment', 'I_infr / Total Investment']
    sizes = [normalized_I_env_opt, normalized_I_infr_opt]
    colors = ['skyblue', 'lightgreen']
    explode = (0.05, 0)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, shadow=True)
    plt.title('Best Solution Normalized Investment Distribution')

    plt.tight_layout()
    plt.show()

    generations = logbook.select("gen")
    min_fitness = logbook.select("min")
    avg_fitness = logbook.select("avg")
    max_fitness = logbook.select("max")

    plt.figure(figsize=(10, 6))
    plt.plot(generations, min_fitness, 'b-', label='Min Fitness')
    plt.plot(generations, avg_fitness, 'g-', label='Avg Fitness')
    plt.plot(generations, max_fitness, 'r-', label='Max Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Fitness Evolution Over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()
