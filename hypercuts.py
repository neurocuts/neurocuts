import math, sys
import datetime

from tree import *


class HyperCuts(object):
    def __init__(self, rules):
        # hyperparameters
        self.leaf_threshold = 16  # number of rules in a leaf
        self.spfac = 4  # space estimation

        # set up
        self.rules = rules

    # HiCuts heuristic to cut a dimeision
    def select_action_hicuts(self, tree, node):
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

    # HyperCuts heuristic to cut a node
    def select_action(self, tree, node):
        # select dimensions
        distinct_components_count = []
        distinct_components_ratio = []
        for i in range(5):
            distinct_components = set()
            for rule in node.rules:
                left = max(rule.ranges[i * 2], node.ranges[i * 2])
                right = min(rule.ranges[i * 2 + 1], node.ranges[i * 2 + 1])
                distinct_components.add((left, right))
            distinct_components_count.append(len(distinct_components))
            distinct_components_ratio.append(
                len(distinct_components) /
                (node.ranges[i * 2 + 1] - node.ranges[i * 2]))
        mean_count = sum(distinct_components_count) / 5.0
        cut_dimensions = [i for i in range(5) \
            if distinct_components_count[i] > mean_count]
        cut_dimensions.sort(key=lambda i: \
            (-distinct_components_count[i], -distinct_components_ratio[i]))

        # compute cuts for the dimensions
        cut_nums = []
        total_cuts = 1
        for i in cut_dimensions:
            range_left = node.ranges[i * 2]
            range_right = node.ranges[i * 2 + 1]
            cut_num = 1
            last_mean = len(node.rules)
            last_max = len(node.rules)
            last_empty = 0
            while True:
                cut_num *= 2

                # compute rule count in each child
                range_per_cut = math.ceil((range_right - range_left) / cut_num)
                child_rules_count = [0 for i in range(cut_num)]
                for rule in node.rules:
                    rule_range_left = max(rule.ranges[i * 2], range_left)
                    rule_range_right = min(rule.ranges[i * 2 + 1], range_right)
                    child_start = (
                        rule_range_left - range_left) // range_per_cut
                    child_end = (
                        rule_range_right - range_left - 1) // range_per_cut
                    for j in range(child_start, child_end + 1):
                        child_rules_count[j] += 1

                # compute statistics
                current_mean = sum(child_rules_count) / len(child_rules_count)
                current_max = max(child_rules_count)
                current_empty = sum([1 for count in child_rules_count \
                    if count == 0])

                # check condition
                if cut_num > range_right - range_left or \
                    total_cuts * cut_num > self.spfac * math.sqrt(len(node.rules)) or \
                    abs(last_mean - current_mean) < 0.1 * last_mean or \
                    abs(last_mean - current_mean) < 0.1 * last_mean or \
                    abs(last_empty - current_empty) > 5:
                    cut_num //= 2
                    break
            cut_nums.append(cut_num)
            total_cuts *= cut_num

        cut_dimensions = [cut_dimensions[i]
            for i in range(len(cut_nums)) \
            if cut_nums[i] != 1]
        cut_nums = [cut_nums[i]
            for i in range(len(cut_nums)) \
            if cut_nums[i] != 1]
        return (cut_dimensions, cut_nums)

    def build_tree(self, rules):

        tree = Tree(
            rules,
            self.leaf_threshold,
            {
                "node_merging": True,
                "rule_overlay": True,
                "region_compaction": True,
                # "rule_pushup"       : True,
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

            # cut_num=0, then turn to hicuts
            cut_dimension, cut_num = self.select_action(tree, node)
            if cut_num == []:
                cut_dimension_hicuts, cut_num_hicuts = self.select_action_hicuts(
                    tree, node)
                if cut_num_hicuts <= 1 and print_count < 100:
                    print("hypercuts turn to hicuts cut_num <=1, node rules number:",\
                        len(node.rules))
                    print_count += 1
                tree.cut_current_node(cut_dimension_hicuts, cut_num_hicuts)
            else:
                tree.cut_current_node_multi_dimension(cut_dimension, cut_num)
            node = tree.get_current_node()
            count += 1
            if count % 10000 == 0:
                print(datetime.datetime.now(), "Depth:", tree.get_depth(),
                      "Remaining nodes:", len(tree.nodes_to_cut))
        # return tree.compute_result()
        tree.result["bytes_per_rule"] = tree.result["bytes_per_rule"] / len(
            tree.rules)
        return tree.result

    def train(self):

        print(datetime.datetime.now(), "Algorithm HyperCuts")

        # divide rules into two sets
        rules_wset = []
        rules_rset = []
        for rule in self.rules:
            if rule.ranges[1] - rule.ranges[0] > 1 and \
                    rule.ranges[3] - rule.ranges[2] > 1:
                rules_wset.append(rule)
            else:
                rules_rset.append(rule)

        # build a tree for each set
        result_wset = self.build_tree(rules_wset)
        result_rset = self.build_tree(rules_rset)
        result = {}
        result[
            "memory_access"] = result_wset["memory_access"] + result_rset["memory_access"]
        result["num_node"] = result_wset["num_node"] + result_rset["num_node"]
        result["bytes_per_rule"] = \
            (result_wset["bytes_per_rule"] * len(rules_wset) + \
            result_rset["bytes_per_rule"] * len(rules_rset)) / \
            len(self.rules)

        print("%s Result %d %d %d" %
              (datetime.datetime.now(), result["memory_access"],
               round(result["bytes_per_rule"]), result["num_node"]))
