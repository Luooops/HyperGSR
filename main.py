import os
import hydra
import torch
import numpy as np
from sklearn.model_selection import KFold

from src.train import train, evaluate_model
from src.experiment import get_run_dir, seed_experiment
from src.plot_utils import plot_adj_matrices
from src.dataset import load_dataset


@hydra.main(version_base="1.3.2", config_path="configs", config_name="experiment")
def main(config):
    torch.cuda.empty_cache()

    # Set random seed for reproducibility (before data loading)
    seed_experiment(config.experiment.kfold.random_state)

    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    kf = KFold(n_splits=config.experiment.kfold.n_splits,
               shuffle=config.experiment.kfold.shuffle,
               random_state=config.experiment.kfold.random_state)

    # Initialize folder structure for this run
    run_dir = get_run_dir(config)

    # Load dataset
    source_data, target_data = load_dataset(config)


    for fold, (train_idx, val_idx) in enumerate(kf.split(source_data)):
        print(f"Training Fold {fold+1}/3")

        # Initialize results directory
        res_dir = f'{run_dir}fold_{fold+1}/'
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # Fetch training and val data for this fold
        source_data_train = [source_data[i] for i in train_idx]
        target_data_train = [target_data[i] for i in train_idx]
        source_data_val = [source_data[i] for i in val_idx]
        target_data_val = [target_data[i] for i in val_idx]

        # Train model for this fold
        train_output = train(
            config=config,
            source_data_train=source_data_train,
            target_data_train=target_data_train,
            source_data_val=source_data_val,
            target_data_val=target_data_val,
            res_dir=res_dir,
        )

        # Evaluate model for this fold (geometry is also passed explicitly).
        eval_output, eval_loss = evaluate_model(
            config=config,
            model=train_output['model'],
            source_data=source_data_val,
            target_data=target_data_val,
            criterion=train_output['criterion'],
            roi_coords_cpu=train_output['roi_coords_cpu'],
        )

        # Final evaluation loss for this fold
        print(f"Final Validation Loss (Target): {eval_loss}")

        # Save source, taregt, and eval output for this fold
        np.save(f'{res_dir}/eval_output.npy', np.array(eval_output))
        np.save(f'{res_dir}/source.npy', np.array([s['mat'] for s in source_data_val]))
        np.save(f'{res_dir}/target.npy', np.array([t['mat'] for t in target_data_val]))


        # Plot predictions for the original fixed sample
        idx = 6
        source_mat_test = source_data_val[idx]['mat']
        target_mat_test = target_data_val[idx]['mat']
        eval_output_t = eval_output[idx]

        plot_adj_matrices(source_mat_test,
                          target_mat_test,
                          eval_output_t,
                          idx,
                          res_dir,
                          file_name=f'eval_sample{idx}')


if __name__ == "__main__":
    main()
