"""
Entry point for all evaluations.
Usage: python scripts/run_eval.py --config configs/default.yaml
                                  --vae_checkpoint checkpoints/best_vae.pt
                                  --policy_checkpoint checkpoints/best_policy.pt
                                  --evals combinatorial planning dag
"""
import os
import sys
import argparse
import yaml
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Run scCausalRL Evaluations")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--vae_checkpoint', type=str, default='checkpoints/phase1_best.pt')
    parser.add_argument('--policy_checkpoint', type=str, default=None)
    parser.add_argument('--evals', nargs='+', default=['combinatorial', 'dag'],
                        choices=['combinatorial', 'planning', 'dag'],
                        help='Which evaluations to run')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--n_dag_samples', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.n_dag_samples is not None:
        config['eval']['n_dag_samples'] = args.n_dag_samples

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluations: {args.evals}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Device: {device}")

    if not os.path.exists(args.vae_checkpoint):
        print(f"ERROR: VAE checkpoint not found: {args.vae_checkpoint}")
        sys.exit(1)

    os.makedirs('logs', exist_ok=True)
    all_results = {}

    # Load Norman data (needed by most evals)
    print("\nLoading Norman dataset...")
    from data.load_norman import load_norman
    data_dict = load_norman(
        path=config['data']['norman_path'],
        n_hvg=config['data']['n_hvg']
    )
    print(f"Loaded: {data_dict['adata'].shape[0]} cells, {len(data_dict['gene_names'])} genes")

    # ============================================================
    # Combinatorial evaluation
    # ============================================================
    if 'combinatorial' in args.evals:
        print("\n" + "=" * 60)
        print("EVALUATION: Combinatorial Perturbation Prediction")
        print("=" * 60)
        from evaluation.eval_combinatorial import eval_combinatorial
        combo_results = eval_combinatorial(
            config=config,
            vae_path=args.vae_checkpoint,
            data_dict=data_dict,
            device=device,
            n_samples=config['eval']['n_dag_samples'],
        )
        all_results['combinatorial'] = {
            k: v for k, v in combo_results.items() if k != 'per_combo'
        }

    # ============================================================
    # DAG evaluation
    # ============================================================
    if 'dag' in args.evals:
        print("\n" + "=" * 60)
        print("EVALUATION: DAG Quality vs CollecTRI")
        print("=" * 60)
        from evaluation.eval_dag import eval_dag
        dag_results = eval_dag(
            config=config,
            vae_path=args.vae_checkpoint,
            data_dict=data_dict,
            device=device,
            output_dir='logs/dag_eval',
        )
        all_results['dag'] = dag_results

    # ============================================================
    # Planning evaluation
    # ============================================================
    if 'planning' in args.evals:
        print("\n" + "=" * 60)
        print("EVALUATION: Multi-step Planning")
        print("=" * 60)
        from evaluation.eval_planning import eval_planning
        plan_results = eval_planning(
            config=config,
            vae_path=args.vae_checkpoint,
            policy_path=args.policy_checkpoint,
            data_dict=data_dict,
            device=device,
            n_dag_samples=config['eval']['n_dag_samples'],
        )
        all_results['planning'] = plan_results

    # ============================================================
    # Print summary and save results
    # ============================================================
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    def _print_nested(d, indent=0):
        for k, v in d.items():
            if isinstance(v, dict):
                print(" " * indent + f"{k}:")
                _print_nested(v, indent + 2)
            elif isinstance(v, float):
                print(" " * indent + f"{k}: {v:.4f}")
            else:
                print(" " * indent + f"{k}: {v}")

    _print_nested(all_results)

    # Save results
    results_path = 'logs/eval_results.json'
    try:
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")
    except Exception as e:
        print(f"Could not save JSON results: {e}")

    return all_results


if __name__ == '__main__':
    main()
