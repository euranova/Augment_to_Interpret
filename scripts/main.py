"""
Main entry point of the augment_to_interpret library.
Train and save a model.
"""

import argparse
import csv
import os
import time
import warnings
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from augment_to_interpret.architectures.contrastive_model import ContrastiveModel
from augment_to_interpret.architectures.my_trainer import MyTrainer
from augment_to_interpret.basic_utils import C, set_seed, get_device
from augment_to_interpret.basic_utils.constants import mkdir_and_return
from augment_to_interpret.complex_utils.model_utils import load_models, save_model
from augment_to_interpret.complex_utils.param_functions import get_model_config, get_result_dir
from augment_to_interpret.datasets import get_task, get_data_loaders, load_node_dataset
from augment_to_interpret.metrics.embedding_evaluation import (
    GSATEvaluator, EmbeddingEvaluation, RegressionEvaluator)

if __name__ == "__main__":
    print("MAIN")
    parser = argparse.ArgumentParser(description='Launch Experiments')
    parser.add_argument('--dataset', type=str, help='dataset used', default="ba_2motifs")
    parser.add_argument('--model_name', type=str, help='name backbone model used', default="GIN")
    parser.add_argument('--loss', type=str, help='training loss', default="simclr_double_aug_info_negative")
    parser.add_argument('--hidden_size', type=int, help='hidden size of backbone model', default=64)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--n_layers', type=int, help='number of layers of backbone model', default=3)
    parser.add_argument('--dropout', type=float, help='dropout ration', default=0.3)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=1000)
    parser.add_argument('--n_clusters', type=int, help='number clusters for evaluation', default=None)
    parser.add_argument('--seed', type=int, help='select seed', default=0)
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=0)
    parser.add_argument('--path_results', type=str, help='results root path',
                        default=Path(C.PATH_RESULTS, "main"))
    parser.add_argument('--use_features_selector', type=int, help='use_features_selector : 1 for yes, 0 for no',
                        default=0)
    parser.add_argument('--use_watchman', type=int, help='use watchman or not : 1 for yes, 0 for no', default=0)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--learning_rate_watchman', type=float, help='learning rate for watchman', default=1e-5)

    parser.add_argument('--watchman_eigenvalues_nb', type=int,
                        help='watchman tries to predict this many eigenvalues of laplacian', default=10)
    parser.add_argument('--watchman_lambda', type=float, help='watchman loss weight', default=.3)
    parser.add_argument('--hidden_dim_watchman', type=int, help='hidden dim of watchman', default=32)

    parser.add_argument('--loss_weight', type=float, help="weight of the infos loss term (for now) if applicable",
                        default=1.0)
    parser.add_argument('--r_info_loss', type=float, help="GSAT r parameter for the info loss", default=0.7)
    parser.add_argument('--temperature_edge_sampling', type=float,
                        help="temperature for the sampling of edges during the training", default=0.7)

    args = parser.parse_args()
    print("PWD", os.getcwd())
    print("FULL ARGS", args)

    dataset_name = args.dataset
    epochs = args.epochs
    seed = args.seed
    cuda_id = args.cuda
    hidden_size = args.hidden_size
    n_layers = args.n_layers
    dropout_ratio = args.dropout
    model_name = args.model_name
    n_clusters = args.n_clusters
    batch_size = args.batch_size
    loss_type = args.loss
    loss_weight = args.loss_weight
    use_features_selector = args.use_features_selector
    use_features_selector = bool(use_features_selector)

    use_watchman = bool(args.use_watchman)
    learning_rate = args.learning_rate
    learning_rate_watchman = args.learning_rate_watchman
    watchman_eigenvalues_nb = args.watchman_eigenvalues_nb
    watchman_lambda = args.watchman_lambda
    hidden_dim_watchman = args.hidden_dim_watchman
    r_info_loss = args.r_info_loss
    temperature_edge_sampling = args.temperature_edge_sampling
    task = get_task(dataset_name)
    path_results = get_result_dir(args.path_results, args)
    assert not (("node" in task) and use_watchman), "Watchman must not be used with node classification task."
    device, cuda_id = get_device(cuda_id)
    print(device)
    set_seed(seed, device)

    print("Saving in", path_results)
    print("Loading data...")
    if "graph" in task:
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(
            C.PATH_DATA,
            dataset_name,
            batch_size=batch_size,
            random_state=seed,
            splits={
                'train': 0.8,
                'valid': 0.1,
                'test': 0.1
            },
        )
    else:
        dataloader, x_dim, edge_attr_dim, num_class, aux_info = load_node_dataset(
            dataset_name, batch_size)

    if n_clusters is None:
        n_clusters = num_class if num_class is not None else 10
    model_config = get_model_config(
        model_name, dataset_name,
        hidden_size, n_layers, dropout_ratio, aux_info=aux_info,
    )
    if "graph" in task:
        model_config['deg'] = aux_info['deg']

    print(f'Features dimension {x_dim}')

    # --------------------------- MODEL COMPONENTS --------------------------- #

    # clf = GIN model
    # extractor = GNN encoder to produce embeddings
    clf, extractor, watchman, optimizer, criterion, features_extractor, optimizer_features = load_models(
        x_dim, edge_attr_dim, num_class, aux_info, model_config, device, task,
        learning_rate, learning_rate_watchman, watchman_eigenvalues_nb, hidden_dim_watchman, use_watchman,
        use_features_selector
    )

    # Embedding evaluator (downstream, not used in training)
    base_model = (LinearSVC(dual=False, fit_intercept=True) if "classification" in task
                  else Ridge(fit_intercept=True, normalize=True, copy_X=True, max_iter=5000))
    evaluator_class = GSATEvaluator() if "classification" in task else RegressionEvaluator()
    evaluator = EmbeddingEvaluation(
        base_model=base_model,
        evaluator=evaluator_class,
        task=task,
        num_tasks=1,
        device=device,
        param_search=True)

    print(f'>>> {loss_type} seed: {seed}')

    summary_path = Path(path_results, "TF_ours")
    writer = SummaryWriter(summary_path)
    path_best_interp_model = mkdir_and_return(path_results, "best_interp_model")
    path_best_clf_model = mkdir_and_return(path_results, "best_clf_model")
    path_best_downstream_clf_model = mkdir_and_return(path_results, "best_downstream_clf_model")
    path_best_silhouette_model = mkdir_and_return(path_results, "best_silhouette_model")
    path_final_model = mkdir_and_return(path_results, "final_model")

    # Save arguments in the same location
    with open(Path(path_results, 'full_args.tsv'), 'w') as csv_file:
        argswriter = csv.writer(csv_file, delimiter='\t')
        for key, value in vars(args).items(): argswriter.writerow([key, value])

    gsat = ContrastiveModel(clf,
                            extractor,
                            criterion,
                            optimizer,
                            optimizer_features=optimizer_features,
                            watchman=watchman,
                            task=task,
                            loss_type=loss_type,
                            learn_edge_att="node" in task,
                            use_watchman=use_watchman,
                            final_r=r_info_loss,
                            temperature_edge_sampling=temperature_edge_sampling,
                            w_info_loss=loss_weight,
                            top_eigenvalues_nb=watchman_eigenvalues_nb,
                            watchman_lambda=watchman_lambda,
                            features_extractor=features_extractor)

    trainer = MyTrainer(gsat, dataset_name=dataset_name, task=task, use_watchman=use_watchman)
    best_interp_acc, best_clf_acc, best_downstream_clf_acc, best_silhouette = (-float("inf") for _ in range(4))

    # The trainer is passed the "gsat" and the evaluator
    for epoch in tqdm(range(epochs)):

        s = time.time()

        if 'graph' in task:
            result_dic_train, all_embs_train, all_clf_labels_train, all_features_att_train, all_features_exp_label_train = trainer.run_one_epoch_and_evaluate_downstream(
                loaders['train'], epoch, 'train', dataset_name, seed, model_config['use_edge_attr'],
                aux_info['multi_label'], writer, device)
            result_dic_valid, all_embs_valid, all_clf_labels_valid, all_features_att_valid, all_features_exp_label_valid = trainer.run_one_epoch_and_evaluate_downstream(
                loaders['valid'], epoch, 'valid', dataset_name, seed, model_config['use_edge_attr'],
                aux_info['multi_label'], writer, device)
            result_dic_test, all_embs_test, all_clf_labels_test, all_features_att_test, all_features_exp_label_test = trainer.run_one_epoch_and_evaluate_downstream(
                loaders['test'], epoch, 'test', dataset_name, seed, model_config['use_edge_attr'],
                aux_info['multi_label'], writer, device)
        else:
            result_dic_train, all_embs_train, all_clf_labels_train, all_features_att_train, all_features_exp_label_train = trainer.run_one_epoch_and_evaluate_downstream(
                dataloader, epoch, 'train', dataset_name, seed, model_config['use_edge_attr'], aux_info['multi_label'],
                writer, device)
            result_dic_valid, all_embs_valid, all_clf_labels_valid, all_features_att_valid, all_features_exp_label_valid = trainer.run_one_epoch_and_evaluate_downstream(
                dataloader, epoch, 'valid', dataset_name, seed, model_config['use_edge_attr'], aux_info['multi_label'],
                writer, device)
            result_dic_test, all_embs_test, all_clf_labels_test, all_features_att_test, all_features_exp_label_test = trainer.run_one_epoch_and_evaluate_downstream(
                dataloader, epoch, 'test', dataset_name, seed, model_config['use_edge_attr'], aux_info['multi_label'],
                writer, device)

        e = time.time()
        print("One epoch training took in seconds " + str(e - s))

        final_dic = {
            "train": result_dic_train,
            "valid": result_dic_valid,
            "test": result_dic_test,
        }

        s = time.time()

        if "graph" in task:
            train_score, val_score, test_score = evaluator.get_embedding_evaluation_from_loaders(
                clf, loaders['train'], loaders['valid'], loaders['test']
            )
        else:
            train_score, val_score, test_score = evaluator.embedding_evaluation(
                all_embs_train, all_clf_labels_train, all_embs_valid, all_clf_labels_valid, all_embs_test,
                all_clf_labels_test
            )

        e = time.time()
        print("One epoch embedding evaluation in seconds " + str(e - s))

        downstream_score_valid = val_score
        writer.add_scalar('DownstreamClfAcc/train', train_score, epoch)
        final_dic.update({"DownstreamClf_acc_train": train_score})
        writer.add_scalar('DownstreamClfAcc/test', test_score, epoch)
        final_dic.update({"DownstreamClf_acc_test": test_score})
        writer.add_scalar('DownstreamClfAcc/valid', val_score, epoch)
        final_dic.update({"DownstreamClf_acc_valid": val_score})
        final_dic.update({"DownstreamClf_epoch": epoch})

        if "classification" in task:
            s = time.time()
            unsup_result_dic = evaluator.unsupervised_scores(all_embs_train,
                                                             all_clf_labels_train,
                                                             all_embs_test,
                                                             all_clf_labels_test,
                                                             n_clusters=n_clusters)
            e = time.time()
            print("One epoch unsupervised scored in seconds " + str(e - s))

            writer.add_scalar('Silhouette/train', unsup_result_dic['train_sil'], epoch)
            writer.add_scalar('ARI/train', unsup_result_dic['train_ari'], epoch)
            writer.add_scalar('Silhouette/test', unsup_result_dic['test_sil'], epoch)
            writer.add_scalar('ARI/test', unsup_result_dic['test_ari'], epoch)
            final_dic = {**final_dic, **unsup_result_dic}

            if unsup_result_dic['train_sil'] > best_silhouette:
                best_silhouette = unsup_result_dic['train_sil']
                save_model(path_best_silhouette_model, gsat, final_dic, "best_silhouette_model")

        if result_dic_valid["Interp_acc"] > best_interp_acc:
            best_interp_acc = result_dic_valid["Interp_acc"]
            save_model(path_best_interp_model, gsat, final_dic, "best_interp_model")
        if result_dic_valid["Clf_acc"] > best_clf_acc:
            best_clf_acc = result_dic_valid["Clf_acc"]
            save_model(path_best_clf_model, gsat, final_dic, "best_clf_model")
        if downstream_score_valid >= best_downstream_clf_acc:
            best_downstream_clf_acc = downstream_score_valid
            save_model(path_best_downstream_clf_model, gsat, final_dic, "best_downstream_clf_model")
