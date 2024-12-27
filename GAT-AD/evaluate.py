from concurrent.futures import ProcessPoolExecutor

import torch


def calculate_gnnet_scores(train_y, train_y_hat, test_y, test_y_hat):
    absolute_error = torch.abs(test_y - test_y_hat)
    mae_training = torch.mean(torch.abs(train_y - train_y_hat), dim=1)
    mae_training_expanded = mae_training.view(-1, 1).expand_as(absolute_error)
    scores = absolute_error / mae_training_expanded
    return scores


def calculate_scores(model_y, model_y_hat, metric="MAE", eps=1e-1):
    if metric == "MAE":
        m = lambda y, y_hat: (y - y_hat).abs()
    elif metric == "MSE":
        m = lambda y, y_hat: (y - y_hat)**2
    else:
        raise NotImplementedError("Uncompatible metric")
    scores = torch.zeros(model_y.shape)
    for flow in range(model_y.shape[0]):
        y_gt = model_y[flow, :]
        y_hat = model_y_hat[flow, :]
        err = m(y_gt, y_hat)
        iqr = err.kthvalue(int(0.75 * err.shape[0])).values - err.kthvalue(int(0.25 * err.shape[0])).values
        flow_score = (err - err.median()) / (iqr.abs() + eps)
        smooth_samples = 5
        smoothed_score = torch.zeros(flow_score.shape)
        smoothed_score[0:smooth_samples-1] = flow_score[0:smooth_samples-1]
        for i in range(smooth_samples, len(flow_score)):
            smoothed_score[i] = flow_score[i-smooth_samples:i+1].mean()
        scores[flow] = smoothed_score

    return scores


def group_detections(detections, T):
    grouped_detections = detections.clone()
    anomaly_start = None
    for i in range(grouped_detections.shape[0]):
        if grouped_detections[i] == True:
            if isinstance(anomaly_start, int) and anomaly_start + T >= i:
                grouped_detections[anomaly_start:i] = torch.ones(grouped_detections[anomaly_start:i].shape).bool()
            anomaly_start = i
    return grouped_detections


def find_intervals(detections):
    intervals = []
    start = None
    for i in range(len(detections)):
        if detections[i] and start is None:
            start = i
        elif not detections[i] and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(detections) - 1))
    return intervals


def calculate_interval_overlap(detections_A, detections_B):
    intervals_A = find_intervals(detections_A)
    overlap = detections_B.clone()
    for start, end in intervals_A:
        if torch.any(detections_B[start:end + 1]):
            overlap[start:end + 1] = True

    return overlap


def calculate_metrics(ground_truth, predictions, return_accuracy=False):
    # Flatten the tensors to calculate metrics
    func_ground_truth = ground_truth.numpy().flatten()
    func_predictions = predictions.numpy().flatten()

    true_positives = (func_predictions * func_ground_truth).sum().item()
    false_positives = (func_predictions * (1 - func_ground_truth)).sum().item()
    false_negatives = ((1 - func_predictions) * func_ground_truth).sum().item()
    true_negatives = ((1 - func_predictions) * (1 - func_ground_truth)).sum().item()

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives + 1e-8)

    # Convert to percentages
    precision *= 100
    recall *= 100
    f1_score *= 100
    accuracy *= 100

    if return_accuracy:
        return precision, recall, f1_score, accuracy
    else:
        return precision, recall, f1_score


# single thread version
# def find_opt_thresh(scores, ground_truth_labels, num_iters=500):
#     optim_topk, global_optim_thresh, global_max_f1 = 0, 0, 0
#     for topk in tqdm(range(1, scores.shape[0] // 2)):
#         topk_scores = scores.topk(dim=0, k=topk).values.sum(dim=0)
#         optim_thresh, max_f1 = 0, 0
#         thresh_values = torch.arange(num_iters) * 1 / num_iters
#         thresh_values = thresh_values * topk_scores.max()
#         for threshold in thresh_values:
#             detections = topk_scores > threshold
#             _, _, f1 = calculate_metrics(
#                 ground_truth=ground_truth_labels,
#                 predictions=detections
#             )
#             if f1 > max_f1:
#                 optim_thresh = threshold
#                 max_f1 = f1
#         if max_f1 > global_max_f1:
#             global_max_f1 = max_f1
#             optim_topk = topk
#             global_optim_thresh = optim_thresh
#         topk_scores = scores.topk(dim=0, k=optim_topk).values.sum(dim=0)
#         optim_detections = topk_scores > global_optim_thresh
#         precision, recall, f1 = calculate_metrics(
#             ground_truth=ground_truth_labels,
#             predictions=optim_detections
#         )
#     return optim_topk, global_optim_thresh, precision, recall, f1


# Function to calculate F1 score for a given threshold
def calculate_f1_for_threshold(args):
    threshold, topk_scores, ground_truth_labels = args
    detections = topk_scores > threshold
    _, _, f1 = calculate_metrics(
        ground_truth=ground_truth_labels,
        predictions=detections
    )
    return threshold, f1


# Function to be parallelized
def find_opt_thresh_for_topk(args):
    topk, scores, ground_truth_labels, num_iters = args
    topk_scores = scores.topk(dim=0, k=topk).values.sum(dim=0)
    thresh_values = torch.arange(num_iters) * 1 / num_iters
    thresh_values *= (topk_scores.max() - topk_scores.min())
    thresh_values -= topk_scores.min().abs()

    # Parallelize the threshold evaluation
    with ProcessPoolExecutor(max_workers=8) as executor:
        threshold_args = [(threshold, topk_scores, ground_truth_labels) for threshold in thresh_values]
        results = list(executor.map(calculate_f1_for_threshold, threshold_args))

    optim_thresh, max_f1 = max(results, key=lambda x: x[1])
    return topk, optim_thresh, max_f1


# Main function
def find_opt_thresh(scores, ground_truth_labels, num_iters=500, topk=None):
    global_max_f1, optim_topk, global_optim_thresh = 0, 0, 0
    num_topk = scores.shape[0] // 2
    results = []

    if topk is None:
        for topk in range(1, num_topk):
            args = (topk, scores, ground_truth_labels, num_iters)
            topk, optim_thresh, max_f1 = find_opt_thresh_for_topk(args)
            results.append((topk, optim_thresh, max_f1))
            if max_f1 > global_max_f1:
                global_max_f1 = max_f1
                optim_topk = topk
                global_optim_thresh = optim_thresh

        # Calculate the final metrics using the optimal topk and threshold
        topk_scores = scores.topk(dim=0, k=optim_topk).values.sum(dim=0)
        optim_detections = topk_scores > global_optim_thresh

    else:
        args = (topk, scores, ground_truth_labels, num_iters)
        topk, optim_thresh, max_f1 = find_opt_thresh_for_topk(args)
        global_optim_thresh = optim_thresh
        topk_scores = scores.topk(dim=0, k=topk).values.sum(dim=0)
        optim_detections = topk_scores > global_optim_thresh
        optim_topk = topk


    precision, recall, f1 = calculate_metrics(
        ground_truth=ground_truth_labels,
        predictions=optim_detections
    )

    return optim_topk, global_optim_thresh, precision, recall, f1


# single thread version
# def find_opt_thresh_2D(scores, ground_truth_labels, num_iters=500):
#     thresh_values = torch.arange(num_iters) * 1 / num_iters
#     thresh_values = thresh_values * (scores.max() - scores.min())
#     thresh_values -= scores.min().abs()
#     max_f1, optim_thresh = 0, 0
#     for threshold in thresh_values:
#         detections = scores > threshold
#         _, _, f1 = calculate_metrics(ground_truth=ground_truth_labels, predictions=detections)
#         if f1 > max_f1:
#             max_f1 = f1
#             optim_thresh = threshold
#     optim_detections = scores > optim_thresh
#     precision, recall, f1 = calculate_metrics(ground_truth=ground_truth_labels, predictions=optim_detections)
#     return optim_thresh, precision, recall, f1


def calculate_f1_for_threshold(args):
    threshold, scores, ground_truth_labels = args
    detections = scores > threshold
    _, _, f1 = calculate_metrics(ground_truth=ground_truth_labels, predictions=detections)
    return threshold, f1


def find_opt_thresh_2D(scores, ground_truth_labels, num_iters=500):
    thresh_values = torch.arange(num_iters) * 1 / num_iters
    thresh_values = thresh_values * (scores.max() - scores.min())
    thresh_values -= scores.min().abs()

    with ProcessPoolExecutor(max_workers=8) as executor:
        threshold_args = [(threshold, scores, ground_truth_labels) for threshold in thresh_values]
        results = list(executor.map(calculate_f1_for_threshold, threshold_args))

    max_f1, optim_thresh = 0, 0
    for threshold, f1 in results:
        if f1 > max_f1:
            max_f1 = f1
            optim_thresh = threshold

    optim_detections = scores > optim_thresh
    precision, recall, f1 = calculate_metrics(ground_truth=ground_truth_labels, predictions=optim_detections)
    return optim_thresh, precision, recall, f1
