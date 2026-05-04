import os
import time
import torch
import joblib

from config import (
    DATASETS_CONFIG,
    MODELS_CONFIG,
    TRAINING_CONFIG,
    PATHS,
    AUGMENTATION_CONFIG,
    ML_CLASSIFIER_CONFIG
)

from data_loader import DataLoader
from approach1_ml_classifier import MLClassifierApproach1
from approach2_end_to_end import EndToEndDLModel


def main():

    print("\n" + "=" * 70)
    print("MULTI DATASET × MULTI MODEL EXPERIMENT")
    print("=" * 70)

    os.makedirs(PATHS['results'], exist_ok=True)
    os.makedirs(PATHS['models'], exist_ok=True)

    all_results = {}

    # =========================
    # LOOP DATASETS
    # =========================
    for dataset_config in DATASETS_CONFIG:

        print("\n" + "#" * 60)
        print(f"DATASET: {dataset_config['name']}")
        print("#" * 60)

        base_config = {
            'DATASET_CONFIG': dataset_config,
            'TRAINING_CONFIG': TRAINING_CONFIG,
            'AUGMENTATION_CONFIG': AUGMENTATION_CONFIG,
            'ML_CLASSIFIER_CONFIG': ML_CLASSIFIER_CONFIG
        }

        # Load data
        data_loader = DataLoader(base_config)
        data = data_loader.load_and_prepare(dataset_name=dataset_config['name'])

        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

        # =========================
        # LOOP MODELS
        # =========================
        for model_config in MODELS_CONFIG:

            print("\n" + "-" * 60)
            print(f"MODEL: {model_config['architecture']}")
            print("-" * 60)

            config = base_config.copy()
            config['MODEL_CONFIG'] = model_config

            exp_name = f"{dataset_config['name']}_{model_config['architecture']}"
            model_dir = os.path.join(PATHS['models'], exp_name)
            os.makedirs(model_dir, exist_ok=True)

            # =========================
            # APPROACH 1
            # =========================
            start = time.time()

            approach1 = MLClassifierApproach1(config)
            approach1.train(X_train, y_train, X_val, y_val)
            a1_result = approach1.evaluate(X_test, y_test)

            a1_result['training_time'] = time.time() - start

            # SAVE APPROACH-1 FEATURE EXTRACTOR (if exists)
            if hasattr(approach1, "feature_extractor"):
                torch.save(
                    approach1.feature_extractor.state_dict(),
                    os.path.join(model_dir, "feature_extractor.pth")
                )

            # SAVE ML MODEL (SVM / LR)
            if hasattr(approach1, "classifier"):
                joblib.dump(
                    approach1.classifier,
                    os.path.join(model_dir, "ml_model.pkl")
                )

            # =========================
            # APPROACH 2
            # =========================
            start = time.time()

            approach2 = EndToEndDLModel(config)
            approach2.build_model()
            approach2.train(X_train, y_train, X_val, y_val)

            a2_result = approach2.evaluate(X_test, y_test)
            a2_result['training_time'] = time.time() - start

            # SAVE APPROACH-2 MODEL
            torch.save(
                approach2.model.state_dict(),
                os.path.join(model_dir, "end_to_end_model.pth")
            )

            # =========================
            # STORE RESULTS
            # =========================
            key = exp_name

            all_results[key] = {
                'approach1': a1_result,
                'approach2': a2_result
            }

            print("\nRESULT:")
            print(f"A1 Acc: {a1_result['accuracy']:.4f}")
            print(f"A2 Acc: {a2_result['accuracy']:.4f}")

    # =========================
    # SAVE SUMMARY
    # =========================
    summary_path = os.path.join(PATHS['results'], 'summary.txt')

    with open(summary_path, 'w') as f:

        f.write("=" * 70 + "\n")
        f.write("FULL EXPERIMENT SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        for key, res in all_results.items():

            f.write(f"{key}\n")
            f.write("-" * 50 + "\n")

            f.write("Approach-1:\n")
            f.write(f"  Acc: {res['approach1']['accuracy']:.4f}\n")
            f.write(f"  F1:  {res['approach1']['f1_score']:.4f}\n")
            f.write(f"  Time: {res['approach1']['training_time']:.2f}s\n\n")

            f.write("Approach-2:\n")
            f.write(f"  Acc: {res['approach2']['accuracy']:.4f}\n")
            f.write(f"  F1:  {res['approach2']['f1_score']:.4f}\n")
            f.write(f"  Time: {res['approach2']['training_time']:.2f}s\n\n")

    print("\n" + "=" * 70)
    print("✓ ALL EXPERIMENTS COMPLETED")
    print(f"✓ Models saved in: {PATHS['models']}")
    print(f"✓ Summary saved in: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()