import numpy as np
from sklearn.linear_model import LinearRegression
from pandarallel import pandarallel

from fit_law import fit_our_law, our_law_transform, fit_vanilla_law, vanilla_law_transform

pandarallel.initialize(progress_bar=False)


def golden_select(data, predict_data_num):
    """
    Select the best model based on the golden data
    """
    data['rank'] = data[predict_data_num].rank(ascending=True)

    best_performance = data[predict_data_num].min()
    worst_performance = data[predict_data_num].max()

    return data['rank'].to_numpy(dtype=np.float32), best_performance, worst_performance, data[predict_data_num].to_numpy(dtype=np.float32)


def zero_shot_select(data, max_data_num, predict_data_num):
    """
    Select the best model based on the zero-shot test loss
    """
    assert 0 in data.columns, "Zero-shot selection requires `0` column in the input data"
    data['rank'] = data[0].rank(ascending=True)

    return data['rank'].to_numpy(dtype=np.float32), data[0].to_numpy(dtype=np.float32)


def subtuning_select(data, max_data_num, predict_data_num):
    """
    Select the best model based on the test loss of finetuning on the largest data num
    """
    data['rank'] = data[max_data_num].rank(ascending=True)

    return data['rank'].to_numpy(dtype=np.float32), data[max_data_num].to_numpy(dtype=np.float32)


def model_size_select(data, max_data_num, predict_data_num):
    """
    Select the best model based on the model size
    """
    assert 'size' in data.columns, "Model size selection requires `size` column in the input data"
    data['rank'] = data['size'].rank(ascending=False)

    return data['rank'].to_numpy(dtype=np.float32), np.log(data['size']).to_numpy(dtype=np.float32)


def _fit_linear_model(data_x, data_y):
    lr = LinearRegression()
    lr.fit(data_x, data_y)

    return lr


def vanilla_fit_select(data, max_data_num, predict_data_num):
    """
    Select the best model based on the vanilla law
    """
    columns = [int(max_data_num / 2**i) for i in range(int(np.log2(max_data_num / 200)), -1, -1)]

    def _func(row):
        train_x = np.log(columns).reshape(-1, 1)
        train_y = np.log(row.to_numpy(dtype=np.float64)).reshape(-1, 1)

        fitted_params, _ = fit_vanilla_law(train_x, train_y)

        test_x = np.log(predict_data_num).reshape(-1, 1)
        pred_y = np.exp(vanilla_law_transform(test_x, *fitted_params))

        if np.isnan(pred_y):
            return 4.
        return pred_y.item()
    
    data['pred'] = data[columns].parallel_apply(_func, axis=1)
    
    data['rank'] = data['pred'].rank(ascending=True)

    return data['rank'].to_numpy(dtype=np.int32), data['pred'].to_numpy(dtype=np.float32)


def our_fit_select(data, max_data_num, predict_data_num):
    """
    Select the best model based on our law
    """
    columns = [0] + [int(max_data_num / 2**i) for i in range(int(np.log2(max_data_num / 200)), -1, -1)]

    def _func(row):
        columns_wo_zero = [1e-10] + columns[1:]
        train_x = np.log(columns_wo_zero)
        train_y = np.log(row.to_numpy(dtype=np.float64))

        params, loss = fit_our_law(train_x, train_y)

        test_x = np.log(predict_data_num)
        pred_y = np.exp(our_law_transform(test_x, *params))
    
        return pred_y.item()
    
    data['pred'] = data[columns].parallel_apply(_func, axis=1)

    data['rank'] = data['pred'].rank(ascending=True)

    return data['rank'].to_numpy(dtype=np.int32), data['pred'].to_numpy(dtype=np.float32)


def ats_select(data, max_data_num, predict_data_num, k=3, delta=5):
    """
    Proposed AtS selection method
    """
    minimum_number = min([i for i in data.columns if isinstance(i, int) and i > 0])
    # assert minimum_number == 200 # DEBUG, need to comment
    columns = [int(max_data_num / 2**i) for i in range(int(np.log2(max_data_num / minimum_number)), -1, -1)]

    def _func(row):
        transition_point = _estimate_phase_transition(row, k=k, delta=delta)
        post_power_phase = columns[transition_point + 1:]
        row = row[post_power_phase]

        train_x = np.log(post_power_phase).reshape(-1, 1)
        train_y = np.log(row.to_numpy(dtype=np.float64)).reshape(-1, 1)

        lr = _fit_linear_model(train_x, train_y)

        test_x = np.log(predict_data_num).reshape(-1, 1)
        pred_y = np.exp(lr.predict(test_x))
    
        return pred_y.item()
    
    # data['pred'] = data[columns].apply(_func, axis=1)
    data['pred'] = data[columns].parallel_apply(_func, axis=1)
    data['rank'] = data['pred'].rank(ascending=True)

    return data['rank'].to_numpy(dtype=np.int32), data['pred'].to_numpy(dtype=np.float32)


def _estimate_phase_transition(row, k=3, delta=5):
    """
    Estimate the phase transition point of the model
    """
    if len(row) <= k:
        return -1

    data_x = row.index.to_numpy(dtype=np.float64).reshape(-1, 1)
    data_y = row.to_numpy(dtype=np.float64).reshape(-1, 1)
    log_data_x = np.log(data_x)
    log_data_y = np.log(data_y)

    for point_to_fit in range(k, len(log_data_x)):
        # use the last k point to fit
        train_x = log_data_x[-point_to_fit:]
        train_y = log_data_y[-point_to_fit:]

        # calculate the residual variance
        lr = _fit_linear_model(train_x, train_y)
        residual = train_y - lr.predict(train_x)
        residual_var = residual.std()

        # calculate the standardized residual
        test_x = log_data_x[-point_to_fit-1:-point_to_fit]
        test_y = log_data_y[-point_to_fit-1:-point_to_fit]
        test_std_residual = np.abs(test_y - lr.predict(test_x)) / residual_var
        
        if test_std_residual > delta:
            break
    
    phase_transition_point = len(log_data_x) - point_to_fit - 1

    return phase_transition_point

