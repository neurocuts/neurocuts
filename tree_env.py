import collections
import time
import os
import numpy as np
import pickle
from gym.spaces import Tuple, Box, Discrete, Dict

from ray.rllib.env import MultiAgentEnv

from tree import Tree, load_rules_from_file
from hicuts import HiCuts

NUM_PART_LEVELS = 6  # 2%, 4%, 8%, 16%, 32%, 64%


class TreeEnv(MultiAgentEnv):
    """NeuroCuts multiagent tree building environment.

    In this env, we aggregate rewards at the end of the episode and
    assign each cut its reward based on the policy performance (actual depth).

    Both modes are modeled as a multi-agent environment. Each "cut" in the tree
    is an action taken by a different agent. All the agents share the same
    policy.
    """

    def __init__(self,
                 rules_file,
                 leaf_threshold=16,
                 max_cuts_per_dimension=5,
                 max_actions_per_episode=5000,
                 max_depth=100,
                 partition_mode=None,
                 reward_shape="linear",
                 depth_weight=1.0,
                 dump_dir=None):

        self.reward_shape = {
            "linear": lambda x: x,
            "log": lambda x: np.log(x),
        }[reward_shape]

        assert partition_mode in [None, "top", "efficuts", "cutsplit"]
        self.partition_enabled = partition_mode == "top"
        if partition_mode in ["efficuts", "cutsplit"]:
            self.force_partition = partition_mode
        else:
            self.force_partition = False

        self.dump_dir = os.path.expanduser(dump_dir)
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

        self.depth_weight = depth_weight
        self.rules_file = rules_file
        self.rules = load_rules_from_file(rules_file)
        self.leaf_threshold = leaf_threshold
        self.max_actions_per_episode = max_actions_per_episode
        self.max_depth = max_depth
        self.num_actions = None
        self.tree = None
        self.node_map = None
        self.child_map = None
        self.max_cuts_per_dimension = max_cuts_per_dimension
        if self.partition_enabled:
            self.num_part_levels = NUM_PART_LEVELS
        else:
            self.num_part_levels = 0
        self.action_space = Tuple([
            Discrete(5),
            Discrete(max_cuts_per_dimension + self.num_part_levels)
        ])
        self.observation_space = Dict({
            "real_obs":
            Box(0, 1, (278, ), dtype=np.float32),
            "action_mask":
            Box(0,
                1, (5 + max_cuts_per_dimension + self.num_part_levels, ),
                dtype=np.float32),
        })

    def reset(self):
        self.num_actions = 0
        self.exceeded_max_depth = []
        self.tree = Tree(
            self.rules,
            self.leaf_threshold,
            refinements={
                "node_merging": True,
                "rule_overlay": True,
                "region_compaction": False,
                "rule_pushup": False,
                "equi_dense": False,
            })
        self.node_map = {
            self.tree.root.id: self.tree.root,
        }
        self.child_map = {}

        if self.force_partition:
            if self.force_partition == "cutsplit":
                self.tree.partition_cutsplit()
            elif self.force_partition == "efficuts":
                self.tree.partition_efficuts()
            else:
                assert False, self.force_partition
            for c in self.tree.root.children:
                self.node_map[c.id] = c
            self.child_map[self.tree.root.id] = [
                c.id for c in self.tree.root.children
            ]

        start = self.tree.current_node
        return {start.id: self._encode_state(start)}

    def step(self, action_dict):
        assert len(action_dict) == 1  # one at a time processing

        new_children = []
        for node_id, action in action_dict.items():
            node = self.node_map[node_id]
            orig_action = action
            if np.isscalar(action):
                assert not self.partition_enabled, action
                partition = False
                cut_dimension = int(action) % 5
                cut_num = int(action) // 5
                action = [cut_dimension, cut_num]
            else:
                if action[1] >= self.max_cuts_per_dimension:
                    assert self.partition_enabled, (
                        action, self.max_cuts_per_dimension)
                    partition = True
                    action[1] -= self.max_cuts_per_dimension
                else:
                    partition = False

            if partition:
                children = self.tree.partition_node(node, action[0], action[1])
            else:
                cut_dimension, cut_num = self.action_tuple_to_cut(node, action)
                children = self.tree.cut_node(node, cut_dimension,
                                              int(cut_num))

            self.num_actions += 1
            num_leaf = 0
            for c in children:
                self.node_map[c.id] = c
                if not self.tree.is_leaf(c):
                    new_children.append(c)
                else:
                    num_leaf += 1
            self.child_map[node_id] = [c.id for c in children]

        node = self.tree.get_current_node()
        while node and (self.tree.is_leaf(node)
                        or node.depth > self.max_depth):
            node = self.tree.get_next_node()
            if node and node.depth > self.max_depth:
                self.exceeded_max_depth.append(node)
        nodes_remaining = self.tree.nodes_to_cut + self.exceeded_max_depth

        obs, rew, done, info = {}, {}, {}, {}

        if (not nodes_remaining
                or self.num_actions > self.max_actions_per_episode
                or self.tree.get_current_node() is None):
            zero_state = self._zeros()
            rew = self.compute_rewards(self.depth_weight)
            obs = {node_id: zero_state for node_id in rew.keys()}
            info = {node_id: {} for node_id in rew.keys()}
            result = self.tree.compute_result()
            rules_remaining = set()
            for n in nodes_remaining:
                for r in n.rules:
                    rules_remaining.add(str(r))
            info[self.tree.root.id] = {
                "bytes_per_rule":
                result["bytes_per_rule"],
                "memory_access":
                result["memory_access"],
                "exceeded_max_depth":
                len(self.exceeded_max_depth),
                "tree_depth":
                self.tree.get_depth(),
                "tree_stats":
                self.tree.get_stats(),
                "tree_stats_str":
                self.tree.stats_str(),
                "nodes_remaining":
                len(nodes_remaining),
                "rules_remaining":
                len(rules_remaining),
                "num_nodes":
                len(self.node_map),
                "useless_fraction":
                float(
                    len([n for n in self.node_map.values()
                         if n.is_useless()])) / len(self.node_map),
                "partition_fraction":
                float(
                    len([
                        n for n in self.node_map.values() if n.is_partition()
                    ])) / len(self.node_map),
                "num_splits":
                self.num_actions,
                "rules_file":
                self.rules_file,
            }
            if not nodes_remaining and self.dump_dir:
                out = os.path.join(
                    self.dump_dir, "tree-{}-{}.pkl".format(
                        result["memory_access"], time.time()))
                with open(out, "wb") as f:
                    pickle.dump(self.tree, f)
            return obs, rew, {"__all__": True}, info

        needs_split = [self.tree.get_current_node()]
        obs.update({s.id: self._encode_state(s) for s in needs_split})
        rew.update({s.id: 0 for s in needs_split})
        done.update({"__all__": False})
        info.update({s.id: {} for s in needs_split})
        return obs, rew, done, info

    def action_tuple_to_cut(self, node, action):
        cut_dimension = action[0]
        range_left = node.ranges[cut_dimension * 2]
        range_right = node.ranges[cut_dimension * 2 + 1]
        cut_num = max(2, min(2**(action[1] + 1), range_right - range_left))
        return (cut_dimension, cut_num)

    def compute_rewards(self, depth_weight):
        depth_to_go = collections.defaultdict(int)
        nodes_to_go = collections.defaultdict(int)
        num_updates = 1
        while num_updates > 0:
            num_updates = 0
            for node_id, node in self.node_map.items():
                if node_id not in depth_to_go:
                    depth_to_go[node_id] = 0
                    nodes_to_go[node_id] = 0
                if node_id in self.child_map:
                    if self.node_map[node_id].is_partition():
                        max_child_depth = sum(
                            [depth_to_go[c] for c in self.child_map[node_id]])
                    else:
                        max_child_depth = 1 + max(
                            [depth_to_go[c] for c in self.child_map[node_id]])
                    if max_child_depth > depth_to_go[node_id]:
                        depth_to_go[node_id] = max_child_depth
                        num_updates += 1
                    sum_child_cuts = len(self.child_map[node_id]) + sum(
                        [nodes_to_go[c] for c in self.child_map[node_id]])
                    if sum_child_cuts > nodes_to_go[node_id]:
                        nodes_to_go[node_id] = sum_child_cuts
                        num_updates += 1
        rew = {
            node_id:
            -depth_weight * self.reward_shape(depth) - (1.0 - depth_weight) *
            self.reward_shape(float(nodes_to_go[node_id]))
            for (node_id, depth) in depth_to_go.items()
            if node_id in self.child_map
        }
        return rew

    def _zeros(self):
        zeros = [0] * 278
        return {
            "real_obs":
            zeros,
            "action_mask":
            [1] * (5 + self.max_cuts_per_dimension + self.num_part_levels),
        }

    def _encode_state(self, node):
        if node.depth > 1:
            action_mask = ([1] * (5 + self.max_cuts_per_dimension) +
                           [0] * self.num_part_levels)
        else:
            assert node.depth == 1, node.depth
            action_mask = ([1] * (5 + self.max_cuts_per_dimension) +
                           [1] * self.num_part_levels)
        return {
            "real_obs": node.get_state(),
            "action_mask": action_mask,
        }
