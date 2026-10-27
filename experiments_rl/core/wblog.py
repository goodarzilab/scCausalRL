"""Tiny, robust W&B logging helper for the RL/planner runs (project CausalBioRL).

Fails soft: if wandb isn't installed/authed (e.g. headless cron), returns None and every call is a no-op,
so runs never break on logging. Usage:
    run = wb_init('briggs_bcppo_g0.5', vars(args), tags=['briggs','bcppo'])
    wb_log(run, {'ppo/reward': r, 'ppo/entropy': h}, step=it)
    wb_summary(run, {'recovery/driver_rate': 0.19}); wb_finish(run)
"""
import os


def wb_init(name, config=None, project='CausalBioRL', tags=None):
    if os.environ.get('WANDB_DISABLED', '').lower() in ('1', 'true'):
        return None
    try:
        import wandb
        return wandb.init(project=project, name=name, config=config or {}, tags=tags, reinit=True)
    except Exception as e:
        print(f"[wandb disabled: {e}]", flush=True)
        return None


def wb_log(run, d, step=None):
    if run is not None:
        try:
            run.log(d, step=step)
        except Exception:
            pass


def wb_summary(run, d):
    if run is not None:
        try:
            run.summary.update(d)
        except Exception:
            pass


def wb_finish(run):
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
