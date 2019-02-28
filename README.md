# NeuroCuts

NeuroCuts is a deep RL algorithm for generating optimized packet classification trees. See the [preprint](https://arxiv.org/abs/1902.10319) for an overview.

## Running NeuroCuts

You can train a NeuroCuts policy for the small `acl5_1k` rule set using the following command. This should converge to an memory access time of 9-10 within 50k timesteps:
```
python run_neurocuts.py --rules=acl5_1k --fast
```

To monitor training progress, open `tensorboard --logdir=~/ray_results` and navigate to the web UI. The important metrics to pay attention to are `rules_remaining_min` (this must reach zero before the policy starts generating "valid" trees), `memory_access_valid_min` (access time metric for valid trees), `bytes_per_rule_valid_min` (bytes per rule metric for valid trees), and `vf_explained_var` (explained variance of the value function, which approaches 1 as the policy converges):

![stats](tensorboard.png)

To kick off a full-scale training run, pass in a comma separated list of rule file names from the `classbench` directory and overrides for other hyperparameters. Example:

```
python run_neurocuts.py --rules=acl1_10k,fw1_10k,ipc1_10k \
    --partition-mode=efficuts \
    --dump-dir=/tmp/trees --num-workers=8 --gpu
```

## Inspecting trees

You can visualize and check the state of generated trees by running `inspect_tree.py <tree.pkl>`. This requires that you specify the `--dump-dir` option when running NeuroCuts training.

## Running baselines

You can run the HiCuts, HyperCuts, EffiCuts, and CutSplit baselines using `run_baselines.py`.
