import math
import datetime

from tree import *


class HiCuts(object):
    def __init__(self, rules):
        # hyperparameters
        self.leaf_threshold = 16  # number of rules in a leaf
        self.spfac = 4  # space estimation

        # set up
        self.rules = rules

    # HiCuts heuristic to cut a dimeision
    def select_action(self, tree, node):
        # select a dimension
        cut_dimension = 0
        max_distinct_components_count = -1
        for i in range(5):
            distinct_components = set()
            for rule in node.rules:
                left = max(rule.ranges[i * 2], node.ranges[i * 2])
                right = min(rule.ranges[i * 2 + 1], node.ranges[i * 2 + 1])
                distinct_components.add((left, right))
            if max_distinct_components_count < len(distinct_components):
                max_distinct_components_count = len(distinct_components)
                cut_dimension = i

        # compute the number of cuts
        range_left = node.ranges[cut_dimension * 2]
        range_right = node.ranges[cut_dimension * 2 + 1]
        #cut_num = min(
        #    max(4, int(math.sqrt(len(node.rules)))),
        #    range_right - range_left)
        cut_num = min(2, range_right - range_left)
        while True:
            sm_C = cut_num
            range_per_cut = math.ceil((range_right - range_left) / cut_num)
            for rule in node.rules:
                rule_range_left = max(rule.ranges[cut_dimension * 2],
                                      range_left)
                rule_range_right = min(rule.ranges[cut_dimension * 2 + 1],
                                       range_right)
                sm_C += (rule_range_right - range_left - 1) // range_per_cut - \
                    (rule_range_left - range_left) // range_per_cut + 1
            if sm_C < self.spfac * len(node.rules) and \
                    cut_num * 2 <= range_right - range_left:
                cut_num *= 2
            else:
                break
        return (cut_dimension, cut_num)

    def train(self):
        print(datetime.datetime.now(), "Algorithm HiCuts")
        tree = self.build_tree()

        result = tree.compute_result()
        result["bytes_per_rule"] = result["bytes_per_rule"] / len(tree.rules)
        print("------mem_result-----")
        print("%s Result %d %d %d" %
              (datetime.datetime.now(), result["memory_access"],
               round(result["bytes_per_rule"]), result["num_node"]))
        # print("------traverse_result-----")
        # result = tree.compute_result()
        # print("%s Result %d %d %d" %
        #     (datetime.datetime.now(),
        #     result["memory_access"],
        #     round(result["bytes_per_rule"]),
        #     result["num_node"]))

    def build_tree(self):
        tree = Tree(
            self.rules, self.leaf_threshold, {
                "node_merging": True,
                "rule_overlay": True,
                "region_compaction": False,
                "rule_pushup": False,
                "equi_dense": False
            })
        node = tree.get_current_node()
        count = 0
        print_count = 0
        while not tree.is_finish():
            if tree.is_leaf(node):
                node = tree.get_next_node()
                continue

            cut_dimension, cut_num = self.select_action(tree, node)
            if cut_num <= 1 and print_count < 100:
                print("hicuts cut_num <=1, node rules number:",
                      len(node.rules))
                print_count += 1
            tree.cut_current_node(cut_dimension, cut_num)
            node = tree.get_current_node()
            count += 1
            if count % 10000 == 0:
                print(datetime.datetime.now(), "Depth:", tree.get_depth(),
                      "Remaining nodes:", len(tree.nodes_to_cut))
        return tree

    def get_depth(self):
        return self.build_tree().get_depth()
