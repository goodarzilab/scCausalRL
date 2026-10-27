"""
Entry point for Phase 1: warm-start causal VAE training.
Usage: python scripts/run_phase1.py --config configs/default.yaml
"""
import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Train Causal VAE")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset: norman or replogle')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda or cpu (auto-detect if not specified)')
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Override number of training epochs')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.dataset is not None:
        config['data']['dataset'] = args.dataset
    if args.n_epochs is not None:
        config['phase1']['n_epochs'] = args.n_epochs

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 1: Training Causal VAE on dataset='{config['data']['dataset']}'")
    print(f"Device: {device}")
    if os.environ.get('X20_ANOMALY') == '1':
        torch.autograd.set_detect_anomaly(True)
        print("[debug] autograd anomaly detection ON (pinpoints NaN-producing op)")

    from training.train_vae import train_vae
    model, data_dict = train_vae(config, device=device)

    print("\nPhase 1 complete.")
    print("  Best checkpoint: checkpoints/phase1_best.pt")
    print("  Training log:   logs/phase1_training.csv")


if __name__ == '__main__':
    main()
