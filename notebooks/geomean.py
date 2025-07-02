from fastfusion.accelerated_imports import np

def combined_sort_idx(data_x, baseline_x):
    combined_x = np.concatenate((data_x, baseline_x))

    data_idx = np.stack((np.ones_like(data_x, dtype=int), np.zeros_like(data_x, dtype=int)))
    baseline_idx = np.stack((np.zeros_like(baseline_x, dtype=int), np.ones_like(baseline_x, dtype=int)))
    combined_idx = np.concatenate((data_idx, baseline_idx), axis=1)

    argsort = np.argsort(combined_x)
    sorted_idx = combined_idx[:,argsort]
    sorted_idx = np.cumsum(sorted_idx, axis=1)

    # Start only after both data and baseline exist
    mask = np.all(sorted_idx > 0, axis=0)
    sorted_idx = sorted_idx[:,mask] - 1

    return sorted_idx
def get_x_intervals(data_x, baseline_x, sorted_idx):
    combined_x = np.max(np.stack([
        data_x[sorted_idx[0,:]],
        baseline_x[sorted_idx[1,:]]
    ]), axis=0)
    x_intervals = combined_x[1:] - combined_x[:-1]
    return x_intervals
def get_y_vals(data_y, baseline_y, sorted_idx):
    combined_y = np.stack([
        data_y[sorted_idx[0,:]],
        baseline_y[sorted_idx[1,:]]
    ])
    return combined_y[:,:-1]

def continuous_gm(data, baseline):
    data_x = data[0,:]
    baseline_x = baseline[0,:]
    sorted_idx = combined_sort_idx(data_x, baseline_x)
    x_intervals = get_x_intervals(data_x, baseline_x, sorted_idx)

    data_y = data[1,:]
    baseline_y = baseline[1,:]
    y_vals = get_y_vals(data_y, baseline_y, sorted_idx)
    y_log = np.log(y_vals[0,:]/y_vals[1,:])

    areas = y_log*x_intervals
    total_area = np.sum(areas)
    average_y_delta = total_area/np.sum(x_intervals)
    
    max_y_delta = np.max(y_log)
    
    # Include the endpoints
    max_y_delta = np.max([max_y_delta, np.log(data_y[-1]/baseline_y[-1])])

    return np.exp(average_y_delta), np.exp(max_y_delta)