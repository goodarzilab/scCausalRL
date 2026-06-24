"""Shared core for the experiments_rl RL/planning methods.

One frozen causal-VAE world model, one shift-native mutilation do-operator, one
continuous-latent goal-conditioned env + penalized reward, and one data-grounded
validation harness — imported by every rlN_* method so the cross-method value-add
ladder is a fair comparison (identical dynamics by construction).
"""
