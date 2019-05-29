import os
from neurocuts import *

import ray
from ray.tune import *


def run_neurocuts(config, reporter):
    random.seed(1)
    rules = load_rules_from_file(config["rules"])
    neuro_cuts = NeuroCuts(
        rules,
        gamma=config["gamma"],
        reporter=reporter,
        onehot_state=config["onehot_state"],
        penalty=config["penalty"])
    neuro_cuts.train()


if __name__ == "__main__":
    ray.init()
    run_experiments({
        "neurocuts-easy-adaptive-bignet": {
            "run": run_neurocuts,
            "config": {
                "rules":
                grid_search([
                    #                    os.path.abspath("classbench/acl1_100"),
                    os.path.abspath("classbench/acl1_200"),
                    os.path.abspath("classbench/acl1_500"),
                    os.path.abspath("classbench/acl1_1000"),
                    os.path.abspath("classbench/acl1_10K"),
                    #                    os.path.abspath("classbench/acl1_100K"),
                ]),
                "penalty":
                grid_search([False]),
                "gamma":
                grid_search([0.99]),
                "onehot_state":
                grid_search([True]),
            },
        },
    })
