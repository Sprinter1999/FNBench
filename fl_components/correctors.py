import numpy as np
import operator
import torch

# To support Joint Optim and SELFIE


class JointOptimCorrector:
    def __init__(self, queue_size, num_classes, data_size):
        self.queue_size = queue_size
        self.num_classes = num_classes

        self.id_2_data_index = {}
        self.counts = np.zeros(data_size, dtype=int)
        # probability histories of samples
        self.probability_history = torch.zeros(data_size, queue_size, num_classes)

        # labels
        # self.hard_labels = torch.zeros(data_size, dtype=torch.int64)
        self.hard_labels = torch.zeros(data_size, num_classes)
        self.soft_labels = torch.zeros(data_size, num_classes)

    def get_data_indices(self, ids):
        data_indices = []

        next_index = max(self.id_2_data_index.values()) + 1 \
            if self.id_2_data_index \
            else 0

        for _id in ids:
            if _id in self.id_2_data_index:
                data_idx = self.id_2_data_index[_id]
            else:
                data_idx = next_index
                self.id_2_data_index[_id] = data_idx
                next_index += 1

            data_indices.append(data_idx)

        return data_indices

    def get_labels(self, ids, labels):
        data_indices = self.get_data_indices(ids)

        init_indices = np.where(self.counts[data_indices] == 0)[0]
        if len(init_indices):
            self.hard_labels[data_indices, labels] = 1.
            self.soft_labels[data_indices, labels] = 1.

        hard_labels = self.hard_labels[data_indices]
        soft_labels = self.soft_labels[data_indices]

        return hard_labels, soft_labels

    def update_probability_history(self, ids, probs):
        data_indices = self.get_data_indices(ids)

        curr_index = self.counts[data_indices] % self.queue_size
        self.probability_history[data_indices, curr_index] = probs
        self.counts[data_indices] += 1

    def update_labels(self):
        self.soft_labels = self.probability_history.mean(dim=1)
        h_labels = torch.argmax(self.soft_labels, dim=1).reshape(-1, 1)
        self.hard_labels = torch.zeros_like(self.soft_labels).scatter_(1, h_labels, 1.)

    def clear_data(self):
        self.probability_history.clear()


class SelfieCorrector:
    def __init__(self, queue_size, uncertainty_threshold, noise_rate, num_classes):
        self.queue_size = queue_size
        self.threshold = uncertainty_threshold
        self.noise_rate = noise_rate

        # prediction histories of samples
        self.prediction_history = {}

        # Max Correctablility / predictive uncertainty
        self.max_certainty = -np.log(1.0 / float(num_classes))

        # Corrected label map
        self.corrected_labels = {}
        self.counts = {}

    def init_id_data(self, _id):
        if _id in self.prediction_history:
            # data for id already initialized
            return

        self.prediction_history[_id] = np.zeros(self.queue_size, dtype=int)
        self.counts[_id] = 0
        self.corrected_labels[_id] = -1

    def update_prediction_history(self, ids, outputs):
        for _id, output in zip(ids, outputs):
            self.init_id_data(_id)

            # append the predicted label to prediction history
            pred_label = np.argmax(output)
            curr_index = self.counts[_id] % self.queue_size
            self.prediction_history[_id][curr_index] = pred_label
            self.counts[_id] += 1

    def separate_clean_and_unclean_samples(self, ids, loss_array):
        num_clean_instances = int(np.ceil(float(len(ids)) * (1.0 - self.noise_rate)))

        loss_map = {_id: loss for _id, loss in zip(ids, loss_array)}
        loss_map = dict(sorted(loss_map.items(), key=operator.itemgetter(1), reverse=False))

        clean_batch = list(loss_map.keys())[:num_clean_instances]

        return clean_batch

    def correct_samples(self, ids):
        # check correctability for each sample
        corrected_batch = []
        for _id in ids:
            pred_label_history = self.prediction_history[_id]
            pred_label_2_counts = {}

            for pred_label in pred_label_history:
                if pred_label not in pred_label_2_counts:
                    pred_label_2_counts[pred_label] = 1
                else:
                    pred_label_2_counts[pred_label] += 1

            negative_entropy = 0.0
            mode_label = None  # most common label
            for pred_label, count in pred_label_2_counts.items():
                prob = float(count) / float(self.queue_size)
                negative_entropy += prob * np.log(prob)

                # Update mode label
                if mode_label is None:
                    mode_label = pred_label
                elif count > pred_label_2_counts[mode_label]:
                    mode_label = pred_label

            certainty = -1.0 * negative_entropy / self.max_certainty

            if certainty <= self.threshold:
                self.corrected_labels[_id] = mode_label

            if self.corrected_labels[_id] != -1:
                corrected_batch.append(_id)

        return corrected_batch

    def correct_and_select_certain_samples(self, ids, X, y, clean_batch, corrected_batch):
        id_2_index = {_id: i for i, _id in enumerate(ids)}

        # Correct samples
        for _id in corrected_batch:
            y[id_2_index[_id]] = int(self.corrected_labels[_id])

        # Select high certainty samples
        high_certainty_samples = set(clean_batch) | set(corrected_batch)
        keep_indices = [id_2_index[_id] for _id in high_certainty_samples]

        X = X[keep_indices]
        y = y[keep_indices]

        return X, y

    def patch_clean_with_corrected_sample_batch(self, ids, X, y, loss_array):
        # 1. separate clean and unclean samples
        clean_batch = self.separate_clean_and_unclean_samples(ids, loss_array)
        # 2. get corrected samples
        corrected_batch = self.correct_samples(ids)
        # 3. merging
        X, y = self.correct_and_select_certain_samples(ids, X, y, clean_batch, corrected_batch)
        return X, y, set(clean_batch)

    def clear_predictions(self):
        self.prediction_history.clear()
        self.counts.clear()
