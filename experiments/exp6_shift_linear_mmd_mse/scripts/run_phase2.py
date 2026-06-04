"""
Entry point for Phase 2: joint RL + world model co-training.
Usage: python scripts/run_phase2.py --config configs/default.yaml --vae_checkpoint checkpoints/phase1_best.pt
"""
import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Joint VAE + PPO Training")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--vae_checkpoint', type=str, default='checkpoints/phase1_best.pt')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phase 2: Joint training")
    print(f"  VAE checkpoint: {args.vae_checkpoint}")
    print(f"  Device: {device}")

    if not os.path.exists(args.vae_checkpoint):
        print(f"ERROR: VAE checkpoint not found: {args.vae_checkpoint}")
        print("  Please run Phase 1 first: python scripts/run_phase1.py")
        sys.exit(1)

    from training.train_joint import train_joint
    model, ppo, env = train_joint(config, args.vae_checkpoint, device=device)

    print("\nPhase 2 complete.")
    print("  Policy checkpoint: checkpoints/phase2_policy_final.pt")
    print("  VAE checkpoint:    checkpoints/phase2_vae_final.pt")
    print("  Training log:      logs/phase2_training.csv")


if __name__ == '__main__':
    main()
