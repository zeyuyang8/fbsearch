import argparse
import json
import os
import subprocess

import optuna

parser = argparse.ArgumentParser()
parser.add_argument("--nproc_per_node", type=int, default=2)
parser.add_argument(
    "--output_dir",
    type=str,
    default="runs/scifact/train-on-real/optuna",
)
args = parser.parse_args()


CWD = "/data/users/zy45/fbsource/fbcode/gen_ai/web_search/genx/scripts"


def run_distributed_training(trial_params: dict, trial_number: int) -> float:
    output_dir = args.output_dir

    port = 29500 + (trial_number + 11) * 3 + 7

    cmd_args = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_port={port}",  # Add this line
        "train_scifact.py",
        "--tracker_name=optuna",
        f"--output_dir={output_dir}",
    ]

    # Add trial hyperparameters
    for param_name, param_value in trial_params.items():
        cmd_args.append(f"--{param_name}={param_value}")

    print(f"Trial {trial_number}: {trial_params}")

    try:
        # Run training
        result = subprocess.run(cmd_args, capture_output=True, text=True, cwd=CWD)

        # Check if process succeeded
        if result.returncode != 0:
            print(f"Trial {trial_number} FAILED with return code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return 0.0

        # Read results
        metrics_file = os.path.join(CWD, output_dir, "metrics.json")
        with open(metrics_file) as f:
            metrics = json.load(f)
            recall = metrics["dev"]["recall"]

        print(f"Trial {trial_number} recall: {recall:.4f}")
        return recall

    except Exception as e:
        print(f"Trial {trial_number} failed: {e}")
        return 0.0


def objective(trial: optuna.Trial) -> float:
    trial_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1]
        ),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 10, 30),
        "lr_warmup_steps": trial.suggest_int("lr_warmup_steps", 50, 200),
    }

    return run_distributed_training(trial_params, trial.number)


def save_study(study: optuna.Study, name="scifact_study"):
    # Save best parameters as JSON
    json_path = os.path.join(CWD, args.output_dir, f"{name}_best.json")
    csv_path = os.path.join(CWD, args.output_dir, f"{name}_trials.csv")
    if study.best_trial:
        best_result = {
            "best_value": study.best_trial.value,
            "best_params": study.best_trial.params,
            "trial_number": study.best_trial.number,
        }
        with open(json_path, "w") as f:
            json.dump(best_result, f, indent=2)

    # Save trials as CSV
    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)

    print(f"Study saved: {json_path}, {csv_path}")


study_name = "scifact_hyperopt"

study = optuna.create_study(
    study_name=study_name,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    load_if_exists=True,  # Resume if exists
)

n_trials = 5
print(f"Starting optimization with {n_trials} trials...")
print(f"Existing trials: {len(study.trials)}")

try:
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best recall: {study.best_trial.value:.4f}")
    print("Best params:", study.best_trial.params)

    # Save results
    save_study(study)

except KeyboardInterrupt:
    print("\nInterrupted! Saving partial results...")
    save_study(study, "scifact_interrupted")
