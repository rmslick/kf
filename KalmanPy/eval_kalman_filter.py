import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter
from simulate_model import simulate_system, create_model_parameters

import json

simulation_data = {}


np.random.seed(21)
(A, H, Q, R) = create_model_parameters()

simulation_data["A"] = A.tolist()
simulation_data["H"] = H.tolist()
simulation_data["Q"] = Q.tolist()
simulation_data["R"] = R.tolist()

print("Sensor covariance: ", R)
K = 20
# initial state
x = np.array([0, 0.1, 0, 0.1])
P = 0 * np.eye(4)

simulation_data["x_init"] = x.tolist()
simulation_data["cov_init"] = P.tolist()

(state, meas) = simulate_system(K, x)

simulation_data["simulated_states"] = state.tolist()
simulation_data["simulated_measurements"] = meas.tolist()

kalman_filter = KalmanFilter(A, H, Q, R, x, P)

est_state = np.zeros((K, 4))
est_cov = np.zeros((K, 4, 4))

for k in range(K):
    kalman_filter.predict()
    kalman_filter.update(meas[k, :])
    (x, P) = kalman_filter.get_state()
    est_state[k, :] = x
    est_cov[k, ...] = P
simulation_data["predicted_states"] = est_state.tolist()
simulation_data["predicted_covariances"] = est_cov.tolist()
# Writing to sample.json
sim = json.dumps(simulation_data)
with open("simulation_data.json", "w") as outfile:
    outfile.write(sim)

plt.figure(figsize=(7, 5))
plt.plot(state[:, 0], state[:, 2], '-bo')
plt.plot(est_state[:, 0], est_state[:, 2], '-ko')
plt.plot(meas[:, 0], meas[:, 1], ':rx')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['true state', 'inferred state', 'observed measurement'])
plt.axis('square')
plt.tight_layout(pad=0)
plt.plot()
plt.savefig("eval_kalman.png")