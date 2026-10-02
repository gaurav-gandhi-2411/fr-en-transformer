# Main L4 run audit (derived by scripts/audit_wandb_run.py)

| item | value |
|---|---|
| run state | finished |
| git_sha / dirty | 9524bac73dfeec223421de4a89c08774f8a63451 / False |
| precision / gpu | bf16 / NVIDIA L4 |
| planned_steps / last logged step | 24645 / 24600 |
| final_step (summary) | 24645 |
| epoch_fraction at last logged step | 8.815617676598594 |
| history rows (train / eval) | 492 (492 / 49) |
| resume_count / wait_seconds_total | 0 / 0 |
| train loss first -> last | 9.1870 -> 2.6876 |
| train loss min | 2.6447 @ 24050 |
| train loss last-1000 mean | 2.7410 |
| val_loss min | 2.9041 @ 24500 |
| val_loss last | 2.9041 @ 24500 |
| val_loss >=2 consecutive rises after min | False |
| non-finite loss / grad_norm rows | 0 / 0 |
| grad_skip_count max | 0 |
| optimizer_stepped False rows | 0 |
| loss_scale distinct | [1] |
| runtime hours | 3.5695841301966666 |
| tok/s median | 71952.9797520033 |
| CU used (ESTIMATE) | 5.497159560502867 (runtime_hours x 1.54 CU/h (GG's Colab Pro L4 rate)) |
