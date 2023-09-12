# Original code from https://github.com/hang53/MEGA/blob/351bc3504489afcc15ca891a824f0c9f3b0f199f/LGA_Lib/embedding_evaluation.py

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score

from ..models import BatchNormSwitch


class GSATEvaluator:
    def __init__(self):
        self.num_tasks = 1
        self.eval_metric = 'accuracy'

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'accuracy':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']
            '''
                y_true: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray)
                    and isinstance(y_true, np.ndarray)):
                raise RuntimeError(
                    'Arguments to Evaluator need to be either numpy ndarray or torch tensor'
                )

            if not y_true.shape == y_pred.shape:
                raise RuntimeError(
                    'Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError(
                    'y_true and y_pred mush to 2-dim arrray, {}-dim array given'
                    .format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError(
                    'Number of tasks should be {} but {} given'.format(
                        self.num_tasks, y_true.shape[1]))

            return y_true, y_pred
        else:
            raise ValueError('Undefined eval metric %s ' % self.eval_metric)

    def _eval_accuracy(self, y_true, y_pred):
        '''
            compute Accuracy score averaged across tasks
        '''
        acc_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values

            is_labeled = y_true[:, i] == y_true[:, i]
            acc = accuracy_score(y_true[is_labeled], y_pred[is_labeled])
            acc_list.append(acc)

        return {'accuracy': sum(acc_list) / len(acc_list)}

    def eval(self, input_dict):
        y_true, y_pred = self._parse_and_check_input(input_dict)
        return self._eval_accuracy(y_true, y_pred)


class RegressionEvaluator:
    """ Highly inpired by AD-GCL
    https://github.com/susheels/adgcl/blob/2605ef8f980934c28d545f2556af5cc6ff48ed18/datasets/zinc.py#L107
    """

    def __init__(self):
        self.num_tasks = 1
        self.eval_metric = 'nmae'

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'nmae':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graph, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

            ## check type
            if not (isinstance(y_true, np.ndarray) and isinstance(y_true, np.ndarray)):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks should be {} but {} given'.format(self.num_tasks,
                                                                                      y_true.shape[1]))

            return y_true, y_pred
        else:
            raise ValueError('Undefined eval metric %s ' % self.eval_metric)

    def _eval_nmae(self, y_true, y_pred):
        '''
            compute MAE score averaged across tasks
        '''
        mae_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            mae_list.append(np.absolute(y_true[is_labeled] - y_pred[is_labeled]).mean())

        return {'nmae': -sum(mae_list) / len(mae_list)}

    def eval(self, input_dict):
        y_true, y_pred = self._parse_and_check_input(input_dict)
        return self._eval_nmae(y_true, y_pred)


def get_emb_y(loader, encoder, device, dtype='numpy', is_rand_label=False):
    with BatchNormSwitch.Switch(0):
        x, y = encoder.get_embeddings(loader, device, is_rand_label)
    if dtype == 'numpy':
        return x, y
    elif dtype == 'torch':
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    else:
        raise NotImplementedError


def get_emb_y_2v(loader, view_learner, encoder, encoder_aug, device, dtype='numpy', is_rand_label=False):
    # x, y = encoder.get_embeddings(loader, device, is_rand_label)
    ret = []
    y = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            batch, x, edge_index = data.batch, data.x, data.edge_index
            edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

            if x is None:
                x = torch.ones((batch.shape[0], 1)).to(device)

            edge_logits = view_learner(batch, x, edge_index, None)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

            x_ori = x
            x, _ = encoder.forward(batch, x, edge_index, edge_weight)
            x_aug, _ = encoder_aug.forward(batch, x_ori, edge_index, None, edge_weight)

            x = torch.cat((x, x_aug), dim=1)
            # x = F.normalize(x, p=2, dim=1)

            ret.append(x.cpu().numpy())
            if is_rand_label:
                y.append(data.rand_label.cpu().numpy())
            else:
                y.append(data.y.cpu().numpy())
    ret = np.concatenate(ret, 0)
    y = np.concatenate(y, 0)
    if dtype == 'numpy':
        return ret, y
    elif dtype == 'torch':
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    else:
        raise NotImplementedError


class EmbeddingEvaluation():
    def __init__(self, base_model, evaluator, task, num_tasks, device, params_dict=None, param_search=True,
                 is_rand_label=False):
        self.is_rand_label = is_rand_label
        self.base_model = base_model
        self.evaluator = evaluator
        self.eval_metric = evaluator.eval_metric
        self.task = task
        self.num_tasks = num_tasks
        self.device = device
        self.param_search = param_search
        self.params_dict = params_dict
        if self.eval_metric == 'nrmse':
            self.gscv_scoring_name = 'neg_root_mean_squared_error'
        elif self.eval_metric == 'nmae':
            self.gscv_scoring_name = 'neg_mean_absolute_error'
        elif self.eval_metric == 'rocauc':
            self.gscv_scoring_name = 'roc_auc'
        elif self.eval_metric == 'accuracy':
            self.gscv_scoring_name = 'accuracy'
        else:
            raise ValueError('Undefined grid search scoring for metric %s ' % self.eval_metric)

        self.downstream_model = None

    def scorer(self, y_true, y_raw):
        y_raw = y_raw.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        input_dict = {"y_true": y_true, "y_pred": y_raw}
        score = self.evaluator.eval(input_dict)[self.eval_metric]
        return score

    def ee_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
        if self.param_search:
            params_dict = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
            self.downstream_model = make_pipeline(StandardScaler(),
                                                  GridSearchCV(self.base_model, params_dict, cv=5,
                                                               scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
                                                  )
        else:
            self.downstream_model = make_pipeline(StandardScaler(), self.base_model)

        self.downstream_model.fit(train_emb, np.squeeze(train_y))

        if self.eval_metric == 'accuracy':
            train_raw = self.downstream_model.predict(train_emb)
            val_raw = self.downstream_model.predict(val_emb)
            test_raw = self.downstream_model.predict(test_emb)
        else:
            train_raw = self.downstream_model.predict_proba(train_emb)[:, 1]
            val_raw = self.downstream_model.predict_proba(val_emb)[:, 1]
            test_raw = self.downstream_model.predict_proba(test_emb)[:, 1]

        return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

    def ee_multioutput_binary_classification(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):

        params_dict = {
            'multioutputclassifier__estimator__C': [1e-1, 1e0, 1e1, 1e2]}
        self.downstream_model = make_pipeline(StandardScaler(), MultiOutputClassifier(
            self.base_model, n_jobs=-1))

        if np.isnan(train_y).any():
            print("Has NaNs ... ignoring them")
            train_y = np.nan_to_num(train_y)
        self.downstream_model.fit(train_emb, train_y)

        train_raw = np.transpose([y_pred[:, 1] for y_pred in self.downstream_model.predict_proba(train_emb)])
        val_raw = np.transpose([y_pred[:, 1] for y_pred in self.downstream_model.predict_proba(val_emb)])
        test_raw = np.transpose([y_pred[:, 1] for y_pred in self.downstream_model.predict_proba(test_emb)])

        return train_raw, val_raw, test_raw

    def ee_regression(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
        if self.param_search:
            params_dict = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
            self.downstream_model = GridSearchCV(self.base_model, params_dict, cv=5,
                                                 scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
        else:
            self.downstream_model = self.base_model

        self.downstream_model.fit(train_emb, np.squeeze(train_y))

        train_raw = self.downstream_model.predict(train_emb)
        val_raw = self.downstream_model.predict(val_emb)
        test_raw = self.downstream_model.predict(test_emb)

        return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1), np.expand_dims(test_raw, axis=1)

    def get_embedding_evaluation_from_loaders(self, encoder, train_loader, valid_loader, test_loader):
        encoder.eval()
        train_emb, train_y = get_emb_y(train_loader, encoder, self.device, is_rand_label=self.is_rand_label)
        val_emb, val_y = get_emb_y(valid_loader, encoder, self.device, is_rand_label=self.is_rand_label)
        test_emb, test_y = get_emb_y(test_loader, encoder, self.device, is_rand_label=self.is_rand_label)
        train_score, val_score, test_score = self.embedding_evaluation(train_emb, train_y, val_emb, val_y, test_emb,
                                                                       test_y)

        return train_score, val_score, test_score

    def embedding_evaluation(self, train_emb, train_y, val_emb, val_y, test_emb, test_y):
        if 'classification' in self.task:

            # Reverse potential one hot encoding (if not multilabel)
            if len(train_y.shape) > 1 and train_y.shape[-1] > 1:
                if all(train_y.sum(axis=1) <= 1):  # Check if multilabel only if already 2D
                    train_y = np.argmax(train_y, -1)
                    val_y = np.argmax(val_y, -1)
                    test_y = np.argmax(test_y, -1)

            # only one class provided, can't train classifier
            if len(np.unique(train_y)) < 2:
                raise ValueError("Not enough labels to evaluate the embeddings , unique labels should be > 1")
            if self.num_tasks == 1:
                train_raw, val_raw, test_raw = self.ee_binary_classification(train_emb, train_y,
                                                                             val_emb, val_y,
                                                                             test_emb, test_y)
            elif self.num_tasks > 1:
                train_raw, val_raw, test_raw = self.ee_multioutput_binary_classification(train_emb, train_y,
                                                                                         val_emb, val_y,
                                                                                         test_emb, test_y)
            else:
                raise NotImplementedError
        else:
            if self.num_tasks == 1:
                train_raw, val_raw, test_raw = self.ee_regression(train_emb, train_y, val_emb, val_y, test_emb, test_y)
            else:
                raise NotImplementedError

        train_score = self.scorer(train_y, train_raw)

        val_score = self.scorer(val_y, val_raw)

        test_score = self.scorer(test_y, test_raw)

        return train_score, val_score, test_score

    def embedding_evaluation2v(self, view_learner, encoder, encoder_aug, train_loader, valid_loader, test_loader):
        encoder.eval()
        train_emb, train_y = get_emb_y_2v(train_loader, view_learner, encoder, encoder_aug, self.device,
                                          is_rand_label=self.is_rand_label)
        val_emb, val_y = get_emb_y_2v(valid_loader, view_learner, encoder, encoder_aug, self.device,
                                      is_rand_label=self.is_rand_label)
        test_emb, test_y = get_emb_y_2v(test_loader, view_learner, encoder, encoder_aug, self.device,
                                        is_rand_label=self.is_rand_label)
        if 'classification' in self.task:

            if self.num_tasks == 1:
                train_raw, val_raw, test_raw = self.ee_binary_classification(train_emb, train_y, val_emb, val_y,
                                                                             test_emb,
                                                                             test_y)
            elif self.num_tasks > 1:
                train_raw, val_raw, test_raw = self.ee_multioutput_binary_classification(train_emb, train_y, val_emb,
                                                                                         val_y,
                                                                                         test_emb, test_y)
            else:
                raise NotImplementedError
        else:
            if self.num_tasks == 1:
                train_raw, val_raw, test_raw = self.ee_regression(train_emb, train_y, val_emb, val_y, test_emb, test_y)
            else:
                raise NotImplementedError

        train_score = self.scorer(train_y, train_raw)

        val_score = self.scorer(val_y, val_raw)

        test_score = self.scorer(test_y, test_raw)

        return train_score, val_score, test_score

    def kf_embedding_evaluation(self, encoder, dataset, folds=10, batch_size=128):
        kf_train = []
        kf_val = []
        kf_test = []

        kf = KFold(n_splits=folds, shuffle=True, random_state=None)
        for k_id, (train_val_index, test_index) in enumerate(kf.split(dataset)):
            print("evaluation round: ", k_id)
            test_dataset = [dataset[int(i)] for i in list(test_index)]
            train_index, val_index = train_test_split(train_val_index, test_size=0.2, random_state=None)

            train_dataset = [dataset[int(i)] for i in list(train_index)]
            val_dataset = [dataset[int(i)] for i in list(val_index)]

            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            valid_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            train_score, val_score, test_score = self.embedding_evaluation(encoder, train_loader, valid_loader,
                                                                           test_loader)

            kf_train.append(train_score)
            kf_val.append(val_score)
            kf_test.append(test_score)

        return np.array(kf_train).mean(), np.array(kf_val).mean(), np.array(kf_test).mean()

    def kf_embedding_evaluation2v(self, view_learner, encoder, encoder_aug, dataset, folds=10, batch_size=128):
        kf_train = []
        kf_val = []
        kf_test = []

        kf = KFold(n_splits=folds, shuffle=True, random_state=None)
        for k_id, (train_val_index, test_index) in enumerate(kf.split(dataset)):
            test_dataset = [dataset[int(i)] for i in list(test_index)]
            train_index, val_index = train_test_split(train_val_index, test_size=0.2, random_state=None)

            train_dataset = [dataset[int(i)] for i in list(train_index)]
            val_dataset = [dataset[int(i)] for i in list(val_index)]

            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            valid_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            train_score, val_score, test_score = self.embedding_evaluation2v(view_learner, encoder, encoder_aug,
                                                                             train_loader, valid_loader, test_loader)

            kf_train.append(train_score)
            kf_val.append(val_score)
            kf_test.append(test_score)

        return np.array(kf_train).mean(), np.array(kf_val).mean(), np.array(kf_test).mean()

    def cluster_and_score(self, embeddings, gt, n_clusters=2):
        """Runs Kmeans on given embeddings and returns the unsupervised
        silhouette score and the supervised ari.

        Args:
            embeddings (_type_): _description_
            gt (_type_): _description_
            n_clusters (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        if len(gt.shape) > 1 and gt.shape[-1] > 1:
            if all(gt.sum(axis=-1) <= 1):
                gt = np.argmax(gt, -1)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        labels = kmeans.labels_
        if len(np.unique(labels)) == 1:  # silhouette needs 2 classes at least
            sil_score = 0
        else:
            sil_score = silhouette_score(embeddings.detach().numpy(), labels, metric='euclidean')
        ari_score = adjusted_rand_score(gt.reshape(-1), labels)
        return sil_score, ari_score

    def unsupervised_scores(self, all_embs_train,
                            all_clf_labels_train,
                            all_embs_test,
                            all_clf_labels_test,
                            n_clusters=2):
        """Returns a dictionary with the unsupervised silhouette scores and the
        supervised ari scores computes on the train/testsets.

        Args:
            all_embs_train (_type_): _description_
            all_clf_labels_train (_type_): _description_
            all_embs_test (_type_): _description_
            all_clf_labels_test (_type_): _description_
            n_clusters (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        train_sil, train_ari = self.cluster_and_score(all_embs_train,
                                                      all_clf_labels_train,
                                                      n_clusters=n_clusters)
        test_sil, test_ari = self.cluster_and_score(all_embs_test,
                                                    all_clf_labels_test,
                                                    n_clusters=n_clusters)

        result_dic = {
            'train_sil': train_sil,
            'train_ari': train_ari,
            'test_sil': test_sil,
            'test_ari': test_ari
        }
        return result_dic
