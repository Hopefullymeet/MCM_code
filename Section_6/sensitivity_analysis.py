import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

alpha0, alpha1, alpha2, alpha3 = 0.5, 0.01, 0.02, 0.03
gamma0, gamma1, gamma2 = 0.1, 0.05, 0.02
beta0,  beta1,  beta2  = 100, 1.2, 0.8
theta1, theta2, theta3, theta4 = 0.5, 0.3, 0.2, 0.1
omega0, omega1, omega2 = 20, 0.1, 0.05
psi0,   psi1,   psi2   = 0.05, 0.02, 0.01

TourNum_min, TourNum_max = 800_000, 1_300_000
Iinfr_base = 50
TourNum_ref = 150_000

R_target = 5e5
S_target = 0.8

def emissions(TourNum, Ienv, ClimateIndex=1):
    return alpha0 + alpha1*TourNum - alpha2*Ienv + alpha3*ClimateIndex

def glacier_melting(E, ClimateIndex=1):
    return gamma0 + gamma1*E + gamma2*ClimateIndex

def revenue(TourNum, TaxRate):
    return beta0 * (TourNum ** beta1) * (TaxRate ** beta2)

def income(Iinfr, TourNum):
    if Iinfr < 0:
        return 1e-6
    val = omega0 + omega1 * Iinfr + omega2 * (TourNum / TourNum_ref)
    return max(val, 1e-6)

def unemployment(Iinfr, TourNum):
    return psi0 - psi1*Iinfr + psi2*(TourNum / TourNum_ref)

def satisfaction(Iinfr, TourNum, Inc, Unemp):
    if Inc <= 0:
        return -1e6
    return (theta1*(Iinfr / Iinfr_base)
            - theta2*(TourNum / TourNum_ref)
            + theta3*np.log(Inc)
            - theta4*Unemp)

def calc_objective(a, TourNum, TaxRate, use_penalty=True):
    rev = revenue(TourNum, TaxRate)
    Ienv = a*rev
    Iinfr = (1.0 - a)*rev

    E = emissions(TourNum, Ienv)
    G = glacier_melting(E)
    R = rev
    Inc = income(Iinfr, TourNum)
    Unemp = unemployment(Iinfr, TourNum)
    S = satisfaction(Iinfr, TourNum, Inc, Unemp)

    raw_obj = 0.4*S + 0.3*R - 0.2*E - 0.1*G

    penalty = 0.0
    if use_penalty:
        if R < R_target:
            penalty += (R_target - R)*0.001
        if S < S_target:
            penalty += (S_target - S)*500

    return raw_obj - penalty

a_baseline      = 0.5
TourNum_baseline= 1_000_000
TaxRate_baseline= 0.075
i_infr_prop_baseline = 0.5

baseline_value = calc_objective(a_baseline, TourNum_baseline, TaxRate_baseline)

N = 8

a_vals = np.linspace(0.2, 0.8, N)
tour_vals = np.linspace(TourNum_min, TourNum_max, N)
tax_vals = np.linspace(0.05, 0.10, N)
i_infr_prop_vals = np.linspace(0.2, 0.8, N)

def pct_change(val, base):
    return (val - base)/ base * 100.0

all_X = []
all_Y = []
all_Z = []

X_1 = np.zeros(N)
Y_1 = np.ones(N)*1
Z_1 = np.zeros(N)

for i in range(N):
    a_ = a_vals[i]
    X_1[i] = pct_change(a_, a_baseline)
    obj_new = calc_objective(a_, TourNum_baseline, TaxRate_baseline)
    Z_1[i] = pct_change(obj_new, baseline_value)

all_X.append(X_1)
all_Y.append(Y_1)
all_Z.append(Z_1)

X_2 = np.zeros(N)
Y_2 = np.ones(N)*2
Z_2 = np.zeros(N)

for i in range(N):
    tour_ = tour_vals[i]
    X_2[i] = pct_change(tour_, TourNum_baseline)
    obj_new= calc_objective(a_baseline, tour_, TaxRate_baseline)
    Z_2[i] = pct_change(obj_new, baseline_value)

all_X.append(X_2)
all_Y.append(Y_2)
all_Z.append(Z_2)

X_3 = np.zeros(N)
Y_3 = np.ones(N)*3
Z_3 = np.zeros(N)

for i in range(N):
    tax_ = tax_vals[i]
    X_3[i] = pct_change(tax_, TaxRate_baseline)
    obj_new= calc_objective(a_baseline, TourNum_baseline, tax_)
    Z_3[i] = pct_change(obj_new, baseline_value)

all_X.append(X_3)
all_Y.append(Y_3)
all_Z.append(Z_3)

def calc_objective_infrProp(i_infr_prop, TourNum, TaxRate):
    rev = revenue(TourNum, TaxRate)
    i_infr_ = i_infr_prop * rev
    i_env_  = (1.0 - i_infr_prop)* rev
    E_ = emissions(TourNum, i_env_)
    G_ = glacier_melting(E_)
    R_ = rev
    I_ = income(i_infr_, TourNum)
    U_ = unemployment(Iinfr, TourNum)
    S_ = satisfaction(Iinfr, TourNum, I_, U_)

    raw_obj = 0.4*S_ + 0.3*R_ - 0.2*E_ - 0.1*G_
    penalty = 0.0
    if R_ < R_target:
        penalty += (R_target - R_)*0.001
    if S_ < S_target:
        penalty += (S_target - S_)*500
    return raw_obj - penalty

X_4 = np.zeros(N)
Y_4 = np.ones(N)*4
Z_4 = np.zeros(N)

for i in range(N):
    infr_p_ = i_infr_prop_vals[i]
    X_4[i] = pct_change(infr_p_, i_infr_prop_baseline)
    obj_new= calc_objective_infrProp(infr_p_, TourNum_baseline, TaxRate_baseline)
    Z_4[i] = pct_change(obj_new, baseline_value)

all_X.append(X_4)
all_Y.append(Y_4)
all_Z.append(Z_4)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

cmap = cm.viridis
alpha_fill = 0.4
for plane_idx in range(4):
    x_arr = all_X[plane_idx]
    y_arr = all_Y[plane_idx]
    z_arr = all_Z[plane_idx]

    ax.plot(x_arr, y_arr, z_arr, color='black', lw=1)

    for j in range(N-1):
        verts = [
            (x_arr[j],   y_arr[j],   0),
            (x_arr[j],   y_arr[j],   z_arr[j]),
            (x_arr[j+1], y_arr[j+1], z_arr[j+1]),
            (x_arr[j+1], y_arr[j+1], 0)
        ]
        poly = Poly3DCollection([verts])
        color_val = plane_idx/3.0
        poly.set_color(cmap(color_val))
        poly.set_alpha(alpha_fill)
        ax.add_collection3d(poly)

ax.set_title("Sensitivity Analysis", fontsize=14)
ax.set_xlabel("Input Change (%)", fontsize=12)
ax.set_ylabel("Input Category",   fontsize=12)
ax.set_zlabel("Objective Change (%)", fontsize=12)

ax.set_yticks([1,2,3,4])
ax.set_yticklabels(["I_env","TourNum","TaxRate","I_infr"])

ax.grid(True)

plt.tight_layout()
plt.savefig("sensitivityAnalysis.png")
plt.show()
