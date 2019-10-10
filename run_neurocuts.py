#!/usr/bin/env python

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import argparse
import os
import glob
import json

from gym.spaces import Tuple, Box, Discrete, Dict
import numpy as np

from ray.rllib.models import ModelCatalog
import ray
from ray import tune
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing

from neurocuts_env import NeuroCutsEnv
from mask import PartitionMaskModel

parser = argparse.ArgumentParser()

parser.add_argument("--rules",
    type=lambda expr: [
        os.path.abspath("classbench/{}".format(r)) for r in expr.split(",")],
    default="acl5_1k",
    help="Rules file name or list of rules files separated by comma.")

parser.add_argument(
    "--dump-dir",
    type=str,
    default="/tmp/neurocuts_out",
    help="Dump valid trees to this directory for later inspection.")

parser.add_argument(
    "--fast",
    action="store_true",
    help="Use fast hyperparam configuration for testing in development.")

parser.add_argument(
    "--partition-mode",
    type=str,
    default=None,
    help="Set the partitioner: [None, 'simple', 'efficuts', 'cutsplit'].")

parser.add_argument(
    "--reward-shape",
    type=str,
    default="linear",
    help="Function to use for combining depth and size weights.")

parser.add_argument("--gae-lambda", type=float, default=0.95)

parser.add_argument(
    "--depth-weight",
    type=float,
    default=1.0,
    help="Weight to use for combining depth and size, in [0, 1]")

parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="Number of parallel workers to request from RLlib.")

parser.add_argument(
    "--gpu", action="store_true", help="Whether to tell RLlib to use a GPU.")

parser.add_argument(
    "--redis-address",
    type=str,
    default=None,
    help="Address of existing Ray cluster to connect to.")


def on_episode_end(info):
    """Report tree custom metrics."""

    episode = info["episode"]
    info = episode.last_info_for(0)
    if not info:
        info = episode.last_info_for((0, 0))
    pid = info["rules_file"].split("/")[-1]
    out = os.path.abspath("valid_trees-{}.txt".format(pid))
    if info["nodes_remaining"] == 0:
        info["tree_depth_valid"] = info["tree_depth"]
        info["num_nodes_valid"] = info["num_nodes"]
        info["num_splits_valid"] = info["num_splits"]
        info["bytes_per_rule_valid"] = info["bytes_per_rule"]
        info["memory_access_valid"] = info["memory_access"]
        with open(out, "a") as f:
            f.write(json.dumps(info))
            f.write("\n")
    else:
        info["tree_depth_valid"] = float("nan")
        info["num_nodes_valid"] = float("nan")
        info["num_splits_valid"] = float("nan")
        info["bytes_per_rule_valid"] = float("nan")
        info["memory_access_valid"] = float("nan")
    del info["rules_file"]
    del info["tree_stats"]
    del info["tree_stats_str"]
    episode.custom_metrics.update(info)


def postprocess_gae(info):
    traj = info["post_batch"]
    infos = traj[SampleBatch.INFOS]
    traj[Postprocessing.ADVANTAGES] = np.array(
        [i["__advantage__"] for i in infos])
    traj[Postprocessing.VALUE_TARGETS] = np.array(
        [i["__value_target__"] for i in infos])
#    print("override adv and v targets", traj[Postprocessing.ADVANTAGES],
#          traj[Postprocessing.VALUE_TARGETS])


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(redis_address=args.redis_address)

    register_env(
        "tree_env", lambda env_config: NeuroCutsEnv(
            env_config["rules"],
            max_depth=env_config["max_depth"],
            max_actions_per_episode=env_config["max_actions"],
            dump_dir=env_config["dump_dir"],
            depth_weight=env_config["depth_weight"],
            reward_shape=env_config["reward_shape"],
            partition_mode=env_config["partition_mode"],
            zero_obs=env_config["zero_obs"],
            tree_gae=env_config["tree_gae"],
            tree_gae_gamma=env_config["tree_gae_gamma"],
            tree_gae_lambda=env_config["tree_gae_lambda"]))

    ModelCatalog.register_custom_model("mask", PartitionMaskModel)

    run_experiments({
        "neurocuts_{}".format(args.partition_mode): {
            "run": "PPO",
            "env": "tree_env",
            "stop": {
                "timesteps_total": 100000 if args.fast else 10000000,
            },
            "config": {
                "log_level": "WARN",
                "num_gpus": 0.2 if args.gpu else 0,
                "num_workers": args.num_workers,
                "sgd_minibatch_size": 100 if args.fast else 1000,
                "sample_batch_size": 200 if args.fast else 5000,
                "train_batch_size": 1000 if args.fast else 15000,
                "batch_mode": "complete_episodes",
                "observation_filter": "NoFilter",
                "model": {
                    "custom_model": "mask",
                    "fcnet_hiddens": [512, 512],
                },
                "vf_share_layers": False,
                "entropy_coeff": 0.01,
                "callbacks": {
                    "on_episode_end": tune.function(on_episode_end),
#                    "on_postprocess_traj": tune.function(postprocess_gae),
                },
                "env_config": {
                    "tree_gae": False,
                    "tree_gae_gamma": 1.0,
                    "tree_gae_lambda": grid_search([args.gae_lambda]),
                    "zero_obs": False,
                    "dump_dir": args.dump_dir,
                    "partition_mode": args.partition_mode,
                    "reward_shape": args.reward_shape,
                    "max_depth": 100 if args.fast else 500,
                    "max_actions": 1000 if args.fast else 15000,
                    "depth_weight": args.depth_weight,
                    "rules": grid_search(args.rules),
                },
            },
        },
    })
