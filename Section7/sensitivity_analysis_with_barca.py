import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

TourNum_min, TourNum_max = 800_000, 1_300_000
Iinfr_base = 50
TourNum_ref = 150_000

R_target = 5e5
S_target = 0.8

juno_params = {
    'alpha0': 0.5, 'alpha1': 0.01, 'alpha2': 0.02, 'alpha3': 0.03,
    'gamma0': 0.1, 'gamma1': 0.05, 'gamma2': 0.02,
    'beta0': 100, 'beta1': 1.2, 'beta2': 0.8,
    'theta1': 0.5, 'theta2': 0.3, 'theta3': 0.2, 'theta4': 0.1,
    'omega0': 20, 'omega1': 0.1, 'omega2': 0.05,
    'psi0': 0.05, 'psi1': 0.02, 'psi2': 0.01
}

barcelona_params = {
    'alpha0': 0.6, 'alpha1': 0.015, 'alpha2': 0.025, 'alpha3': 0.035,
    'gamma0': 0.15, 'gamma1': 0.055, 'gamma2': 0.025,
    'beta0': 110, 'beta1': 1.3, 'beta2': 0.85,
    'theta1': 0.6, 'theta2': 0.35, 'theta3': 0.25, 'theta4': 0.15,
    'omega0': 22, 'omega1': 0.12, 'omega2': 0.06,
    'psi0': 0.06, 'psi1': 0.025, 'psi2': 0.015
}

def emissions(TourNum, Ienv, alpha0, alpha1, alpha2, alpha3, ClimateIndex=1):
    return alpha0 + alpha1 * TourNum - alpha2 * Ienv + alpha3 * ClimateIndex

def glacier_melting(E, gamma0, gamma1, gamma2, ClimateIndex=1):
    return gamma0 + gamma1 * E + gamma2 * ClimateIndex

def revenue(TourNum, TaxRate, beta0, beta1, beta2):
    return beta0 * (TourNum ** beta1) * (TaxRate ** beta2)

def income(Iinfr, TourNum, omega0, omega1, omega2):
    if Iinfr < 0:
        return 1e-6
    val = omega0 + omega1 * Iinfr + omega2 * (TourNum / TourNum_ref)
    return max(val, 1e-6)

def unemployment(Iinfr, TourNum, psi0, psi1, psi2):
    return psi0 - psi1 * Iinfr + psi2 * (TourNum / TourNum_ref)

def satisfaction(Iinfr, TourNum, Inc, Unemp, theta1, theta2, theta3, theta4):
    if Inc <= 0:
        return -1e6
    return (theta1 * (Iinfr / Iinfr_base)
            - theta2 * (TourNum / TourNum_ref)
            + theta3 * np.log(Inc)
            - theta4 * Unemp)

def calc_objective(a, TourNum, TaxRate, params, use_penalty=True):
    rev = revenue(TourNum, TaxRate, params['beta0'], params['beta1'], params['beta2'])
    Ienv = a * rev
    Iinfr = (1.0 - a) * rev
    
    E = emissions(TourNum, Ienv, params['alpha0'], params['alpha1'], params['alpha2'], params['alpha3'])
    G = glacier_melting(E, params['gamma0'], params['gamma1'], params['gamma2'])
    R = rev
    Inc = income(Iinfr, TourNum, params['omega0'], params['omega1'], params['omega2'])
    Unemp = unemployment(Iinfr, TourNum, params['psi0'], params['psi1'], params['psi2'])
    S = satisfaction(Iinfr, TourNum, Inc, Unemp, params['theta1'], params['theta2'], params['theta3'], params['theta4'])
    
    raw_obj = 0.4 * S + 0.3 * R - 0.2 * E - 0.1 * G
    
    penalty = 0.0
    if use_penalty:
        if R < R_target:
            penalty += (R_target - R) * 0.001
        if S < S_target:
            penalty += (S_target - S) * 500
    
    return raw_obj - penalty

def sensitivity_analysis(params, a_baseline, TourNum_baseline, TaxRate_baseline, i_infr_prop_baseline, N=8):
    a_vals = np.linspace(0.2, 0.8, N)
    tour_vals = np.linspace(TourNum_min, TourNum_max, N)
    tax_vals = np.linspace(0.05, 0.10, N)
    i_infr_prop_vals = np.linspace(0.2, 0.8, N)
    
    def pct_change(val, base):
        return (val - base) / base * 100.0
    
    baseline_value = calc_objective(a_baseline, TourNum_baseline, TaxRate_baseline, params)
    
    all_X = []
    all_Y = []
    all_Z = []
    
    X_1 = np.zeros(N)
    Y_1 = np.ones(N) * 1
    Z_1 = np.zeros(N)
    for i in range(N):
        a_ = a_vals[i]
        X_1[i] = pct_change(a_, a_baseline)
        obj_new = calc_objective(a_, TourNum_baseline, TaxRate_baseline, params)
        Z_1[i] = pct_change(obj_new, baseline_value)
    all_X.append(X_1)
    all_Y.append(Y_1)
    all_Z.append(Z_1)
    
    X_2 = np.zeros(N)
    Y_2 = np.ones(N) * 2
    Z_2 = np.zeros(N)
    for i in range(N):
        tour_ = tour_vals[i]
        X_2[i] = pct_change(tour_, TourNum_baseline)
        obj_new = calc_objective(a_baseline, tour_, TaxRate_baseline, params)
        Z_2[i] = pct_change(obj_new, baseline_value)
    all_X.append(X_2)
    all_Y.append(Y_2)
    all_Z.append(Z_2)
    
    X_3 = np.zeros(N)
    Y_3 = np.ones(N) * 3
    Z_3 = np.zeros(N)
    for i in range(N):
        tax_ = tax_vals[i]
        X_3[i] = pct_change(tax_, TaxRate_baseline)
        obj_new = calc_objective(a_baseline, TourNum_baseline, tax_, params)
        Z_3[i] = pct_change(obj_new, baseline_value)
    all_X.append(X_3)
    all_Y.append(Y_3)
    all_Z.append(Z_3)
    
    X_4 = np.zeros(N)
    Y_4 = np.ones(N) * 4
    Z_4 = np.zeros(N)
    for i in range(N):
        infr_p_ = i_infr_prop_vals[i]
        X_4[i] = pct_change(infr_p_, i_infr_prop_baseline)
        rev = revenue(TourNum_baseline, TaxRate_baseline, params['beta0'], params['beta1'], params['beta2'])
        Iinfr = (1.0 - infr_p_) * rev
        Ienv = infr_p_ * rev
        E = emissions(TourNum_baseline, Ienv, params['alpha0'], params['alpha1'], params['alpha2'], params['alpha3'])
        G = glacier_melting(E, params['gamma0'], params['gamma1'], params['gamma2'])
        R = rev
        Inc = income(Iinfr, TourNum_baseline, params['omega0'], params['omega1'], params['omega2'])
        Unemp = unemployment(Iinfr, TourNum_baseline, params['psi0'], params['psi1'], params['psi2'])
        S = satisfaction(Iinfr, TourNum_baseline, Inc, Unemp, params['theta1'], params['theta2'], params['theta3'], params['theta4'])
        raw_obj = 0.4 * S + 0.3 * R - 0.2 * E - 0.1 * G
        penalty = 0.0
        if R < R_target:
            penalty += (R_target - R) * 0.001
        if S < S_target:
            penalty += (S_target - S) * 500
        obj_new = raw_obj - penalty
        Z_4[i] = pct_change(obj_new, baseline_value)
    all_X.append(X_4)
    all_Y.append(Y_4)
    all_Z.append(Z_4)
    
    return all_X, all_Y, all_Z

a_baseline = 0.5
TourNum_baseline = 1_000_000
TaxRate_baseline = 0.075
i_infr_prop_baseline = 0.5

all_X_juno, all_Y_juno, all_Z_juno = sensitivity_analysis(
    juno_params, a_baseline, TourNum_baseline, TaxRate_baseline, i_infr_prop_baseline, N=8
)

all_X_barcelona, all_Y_barcelona, all_Z_barcelona = sensitivity_analysis(
    barcelona_params, a_baseline, TourNum_baseline, TaxRate_baseline, i_infr_prop_baseline, N=8
)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

cmap = cm.viridis
colors = {'Juno': 'blue', 'Barcelona': 'green'}
alpha_fill = 0.3

for plane_idx in range(4):
    x_arr = all_X_juno[plane_idx]
    y_arr = all_Y_juno[plane_idx]
    z_arr = all_Z_juno[plane_idx]
    
    ax.plot(x_arr, y_arr, z_arr, color=colors['Juno'], lw=1)
    
    for j in range(len(x_arr)-1):
        verts = [
            (x_arr[j],   y_arr[j],   0),
            (x_arr[j],   y_arr[j],   z_arr[j]),
            (x_arr[j+1], y_arr[j+1], z_arr[j+1]),
            (x_arr[j+1], y_arr[j+1], 0)
        ]
        poly = Poly3DCollection([verts], facecolors=colors['Juno'], alpha=alpha_fill)
        ax.add_collection3d(poly)

for plane_idx in range(4):
    x_arr = all_X_barcelona[plane_idx]
    y_arr = all_Y_barcelona[plane_idx]
    z_arr = all_Z_barcelona[plane_idx]
    
    ax.plot(x_arr, y_arr, z_arr, color=colors['Barcelona'], lw=1)
    
    for j in range(len(x_arr)-1):
        verts = [
            (x_arr[j],   y_arr[j],   0),
            (x_arr[j],   y_arr[j],   z_arr[j]),
            (x_arr[j+1], y_arr[j+1], z_arr[j+1]),
            (x_arr[j+1], y_arr[j+1], 0)
        ]
        poly = Poly3DCollection([verts], facecolors=colors['Barcelona'], alpha=alpha_fill)
        ax.add_collection3d(poly)

ax.set_title("Sensitivity Analysis for Juno and Barcelona", fontsize=16)
ax.set_xlabel("Input Change (%)", fontsize=12)
ax.set_ylabel("Input Category", fontsize=12)
ax.set_zlabel("Objective Change (%)", fontsize=12)

ax.set_yticks([1,2,3,4])
ax.set_yticklabels(["I_env", "TourNum", "TaxRate", "I_infr"])

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors['Juno'], edgecolor='k', label='Juno'),
    Patch(facecolor=colors['Barcelona'], edgecolor='k', label='Barcelona')
]
ax.legend(handles=legend_elements, loc='upper left')

ax.grid(True)

plt.tight_layout()
plt.savefig("sensitivityAnalysis_Juno_Barcelona.png")
plt.show()

all_X_diff = []
all_Y_diff = []
all_Z_diff = []

for plane_idx in range(4):
    x_bar = all_X_barcelona[plane_idx]
    y_bar = all_Y_barcelona[plane_idx]
    z_bar = all_Z_barcelona[plane_idx]

    x_jun = all_X_juno[plane_idx]
    y_jun = all_Y_juno[plane_idx]
    z_jun = all_Z_juno[plane_idx]
    
    z_diff = z_bar - z_jun
    
    all_X_diff.append(x_bar)
    all_Y_diff.append(y_bar)
    all_Z_diff.append(z_diff)

fig_diff = plt.figure(figsize=(10, 8))
ax_diff = fig_diff.add_subplot(111, projection='3d')

scale_factor = 1.0

category_colors = ["red", "gold", "purple", "cyan"]

category_labels = ["I_env (a)", "TourNum", "TaxRate", "I_infr"]

for plane_idx in range(4):
    x_arr = all_X_diff[plane_idx]
    y_arr = all_Y_diff[plane_idx]
    z_arr = all_Z_diff[plane_idx] * scale_factor

    color_ = category_colors[plane_idx]
    
    ax_diff.plot(x_arr, y_arr, z_arr, color=color_, lw=2, 
                 label=category_labels[plane_idx])
    
    for j in range(len(x_arr) - 1):
        verts = [
            (x_arr[j],   y_arr[j],   0),
            (x_arr[j],   y_arr[j],   z_arr[j]),
            (x_arr[j+1], y_arr[j+1], z_arr[j+1]),
            (x_arr[j+1], y_arr[j+1], 0)
        ]
        poly = Poly3DCollection([verts], facecolors=color_, alpha=0.3)
        ax_diff.add_collection3d(poly)

ax_diff.set_title("Difference: (Barcelona - Juno)", fontsize=16)
ax_diff.set_xlabel("Input Change (%)", fontsize=12)
ax_diff.set_ylabel("Input Category", fontsize=12)
ax_diff.set_zlabel("Objective Change Diff (%)", fontsize=12)

ax_diff.set_yticks([1, 2, 3, 4])
ax_diff.set_yticklabels(["I_env (a)", "TourNum", "TaxRate", "I_infr"])

ax_diff.legend(loc="upper left")
ax_diff.grid(True)

plt.savefig("Difference_Barca_Juneau.png")
plt.tight_layout()
plt.show()
