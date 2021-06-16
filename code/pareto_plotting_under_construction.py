## Needs to be copied and pasted back into the notebook for this to work

## Also, needs edits to be made simpler

y_true = output_trainval

CP_arr_1 = np.zeros(10)
AIW_arr_1 = np.zeros(10)

CP_arr_2 = np.zeros(10)
AIW_arr_2 = np.zeros(10)

# Metrics: mean requirement (MW), standard deviation of coverage (%)

requirement_1 = np.zeros([len(tau_arr), len(CV_folds)])
coverage_1 = np.zeros([len(tau_arr), len(CV_folds)])
risk_1 = np.zeros([len(tau_arr), len(CV_folds)])

requirement_2 = np.zeros([len(tau_arr), len(CV_folds)])
coverage_2 = np.zeros([len(tau_arr), len(CV_folds)])
risk_2 = np.zeros([len(tau_arr), len(CV_folds)])

for i, tau in enumerate(tau_arr):

    for j, CV in enumerate(CV_folds):
        y_pred1 = pred_trainval1[(tau, CV)]
        y_pred2 = pred_trainval2[(tau, CV)]

        requirement_1[i, j] = requirement(y_true, y_pred1)
        coverage_1[i, j] = coverage(y_true, y_pred1)
        risk_1[i, j] = pinball_risk(y_true, y_pred1)

        requirement_2[i, j] = requirement(y_true, y_pred2)
        coverage_2[i, j] = coverage(y_true, y_pred2)
        risk_2[i, j] = pinball_risk(y_true, y_pred2)

# Pareto Comparison

mean_AIW_1 = np.mean(requirement_1, axis=1)
std_coverage_1 = np.std(coverage_1, axis=1)
RMS_coverage_1 = np.sqrt(
    np.mean((coverage_1 - np.array(tau_arr).reshape(-1, 1)) ** 2, axis=1)
)

mean_AIW_2 = np.mean(requirement_2, axis=1)
std_coverage_2 = np.std(coverage_2, axis=1)
RMS_coverage_2 = np.sqrt(
    np.mean((coverage_2 - np.array(tau_arr).reshape(-1, 1)) ** 2, axis=1)
)

scale = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
a_ = np.mean(np.abs(np.append(mean_AIW_1, mean_AIW_2)))
b_ = 100 * np.mean(np.append(std_coverage_1, std_coverage_2))
for s in scale:
    a = s * a_
    b = s * b_

    frontier_x = np.linspace(-a, a, 1000)
    frontier_y = (b / a) * np.sqrt(a ** 2 - frontier_x ** 2)

    plt.plot(frontier_x, frontier_y, "k", linewidth=0.5)

plt.scatter(mean_AIW_1, 100 * RMS_coverage_1, alpha=0.7, c="blue", label="v1.1")
plt.scatter(
    mean_AIW_2, 100 * RMS_coverage_2, alpha=0.7, c="orange", label="v1.1 - no calendar"
)
plt.plot(mean_AIW_1, 100 * RMS_coverage_1, "blue", linewidth=0.75)
plt.plot(mean_AIW_2, 100 * RMS_coverage_2, "orange", linewidth=0.75)
plt.legend(loc="upper right")
plt.xlabel("Mean Requirement (MW)")
plt.ylabel("RMSE of Coverage (%)")
plt.axis([-600, 600, 0.0, 2.0])
plt.savefig(
    os.path.join("Pareto Comparison Plots", "no_calendar_comparison.jpg"), dpi=250
)
plt.show()

# Pareto Comparison

mean_AIW_1 = np.mean(requirement_1, axis=1)
std_coverage_1 = np.std(coverage_1, axis=1)
RMS_coverage_1 = np.sqrt(
    np.mean((coverage_1 - np.array(tau_arr).reshape(-1, 1)) ** 2, axis=1)
)

mean_AIW_2 = np.mean(requirement_2, axis=1)
std_coverage_2 = np.std(coverage_2, axis=1)
RMS_coverage_2 = np.sqrt(
    np.mean((coverage_2 - np.array(tau_arr).reshape(-1, 1)) ** 2, axis=1)
)

scale = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
a_ = np.mean(np.abs(np.append(mean_AIW_1, mean_AIW_2)))
b_ = 100 * np.mean(np.append(std_coverage_1, std_coverage_2))
for s in scale:
    a = s * a_
    b = s * b_

    frontier_x = np.linspace(-a, a, 1000)
    frontier_y = (b / a) * np.sqrt(a ** 2 - frontier_x ** 2)

    plt.plot(frontier_x, frontier_y, "k", linewidth=0.5)

plt.scatter(mean_AIW_1, 100 * RMS_coverage_1, alpha=0.7, c="blue", label="v1.1")
plt.scatter(
    mean_AIW_2, 100 * RMS_coverage_2, alpha=0.7, c="orange", label="v1.1 - no calendar"
)
plt.plot(mean_AIW_1, 100 * RMS_coverage_1, "blue", linewidth=0.75)
plt.plot(mean_AIW_2, 100 * RMS_coverage_2, "orange", linewidth=0.75)
plt.legend(loc="upper right")
plt.xlabel("Mean Requirement (MW)")
plt.ylabel("RMSE of Coverage (%)")
plt.axis([-600, 600, 0.0, 2.0])
plt.savefig(
    os.path.join("Pareto Comparison Plots", "no_calendar_comparison.jpg"), dpi=250
)
plt.show()

# Pareto Comparison

mean_risk_1 = np.mean(risk_1, axis=1)
std_risk_1 = np.std(risk_1, axis=1)

mean_risk_2 = np.mean(risk_2, axis=1)
std_risk_2 = np.std(risk_2, axis=1)

mean_AIW_1 = np.mean(requirement_1, axis=1)
std_coverage_1 = np.std(coverage_1, axis=1)

mean_AIW_2 = np.mean(requirement_2, axis=1)
std_coverage_2 = np.std(coverage_2, axis=1)

plt.scatter(mean_risk_1, 100 * std_coverage_1, alpha=0.7, c="blue", label="v1.1")
plt.scatter(
    mean_risk_2, 100 * std_coverage_2, alpha=0.7, c="orange", label="v1.1 - no calendar"
)
plt.plot(mean_risk_1, 100 * std_coverage_1, "blue", linewidth=0.75)
plt.plot(mean_risk_2, 100 * std_coverage_2, "orange", linewidth=0.75)
plt.legend(loc="upper right")
plt.xlabel("Pinball Loss (MW)")
plt.ylabel("Standard Deviation of Coverage (%)")
plt.savefig(
    os.path.join(
        "Pareto Comparison Plots", "no_calendar_comparison_CP_pinball_loss.jpg"
    ),
    dpi=250,
)
plt.show()

# Pareto Comparison

mean_risk_1 = np.mean(risk_1, axis=1)
std_risk_1 = np.std(risk_1, axis=1)

mean_risk_2 = np.mean(risk_2, axis=1)
std_risk_2 = np.std(risk_2, axis=1)

mean_AIW_1 = np.mean(requirement_1, axis=1)
std_coverage_1 = np.std(coverage_1, axis=1)

mean_AIW_2 = np.mean(requirement_2, axis=1)
std_coverage_2 = np.std(coverage_2, axis=1)

plt.scatter(mean_risk_1, std_risk_1, alpha=0.7, c="blue", label="v1.1")
plt.scatter(mean_risk_2, std_risk_2, alpha=0.7, c="orange", label="v1.1 - no calendar")
plt.plot(mean_risk_1, std_risk_1, "blue", linewidth=0.75)
plt.plot(mean_risk_2, std_risk_2, "orange", linewidth=0.75)
plt.legend(loc="upper right")
plt.xlabel("Pinball Loss (MW)")
plt.ylabel("Standard Deviation of Pinball Loss (MW)")
plt.savefig(
    os.path.join(
        "Pareto Comparison Plots", "no_calendar_comparison_CP_pinball_loss.jpg"
    ),
    dpi=250,
)
plt.show()

# Plot to show how standard deviation of CP relates to standard deviation of pinball loss (generalizability metrics)

plt.scatter(std_risk_1, std_coverage_1, alpha=0.7, c="blue")
plt.scatter(std_risk_2, std_coverage_2, alpha=0.7, c="orange")
plt.xlabel("Standard Deviation of Pinball Loss (MW)")
plt.ylabel("Standard Deviation of Coverage (%)")
plt.show()

# Plot to show how standard deviation of CP relates to standard deviation of pinball loss (generalizability metrics)

plt.scatter(std_risk_1, RMS_coverage_1, alpha=0.7, c="blue")
plt.scatter(std_risk_2, RMS_coverage_2, alpha=0.7, c="orange")
plt.xlabel("Standard Deviation of Pinball Loss (MW)")
plt.ylabel("RMSE of Coverage (%)")
plt.show()
