import math, sys
import datetime

from tree import *


class CutSplit(object):
    def __init__(self, rules):
        # hyperparameters
        self.leaf_threshold = 16  # number of rules in a leaf
        self.spfac = 4  # space estimation
        self.ip_threshold = 2**24  # threshold for big ip address
        self.ficut_threshold = 16  # threshold to switch to hypersplit

        # set up
        self.rules = rules

    def separate_rules(self, rules):
        def compute_ip_ranges(rule):
            return (rule.ranges[1] - rule.ranges[0],
                    rule.ranges[3] - rule.ranges[2])

        def update_bins(rule, src_bins, dst_bins, bin_size):
            src_ip_range, dst_ip_range = compute_ip_ranges(rule)
            if src_ip_range < dst_ip_range:
                for i in range(
                        int(rule.ranges[0] // bin_size),
                        int(math.ceil(rule.ranges[1] / bin_size))):
                    src_bins[i] += 1
            else:
                for i in range(
                        int(rule.ranges[2] // bin_size),
                        int(math.ceil(rule.ranges[3] / bin_size))):
                    dst_bins[i] += 1

        # separate rules based on src and dst ip ranges
        # subset 0: big src ip, big dst ip
        # subset 1: small src ip, big dst i
        # subset 2: big src ip, small dst ip
        # subset 3: small src ip, small dst ip
        rule_subsets = [[] for i in range(4)]
        bin_size = 2**12
        src_bins = [0 for i in range(2**20)]
        dst_bins = [0 for i in range(2**20)]
        for rule in rules:
            src_ip_range, dst_ip_range = compute_ip_ranges(rule)

            if src_ip_range > self.ip_threshold and \
                    dst_ip_range > self.ip_threshold:
                rule_subsets[0].append(rule)
                update_bins(rule, src_bins, dst_bins, bin_size)
            elif src_ip_range <= self.ip_threshold and \
                    dst_ip_range > self.ip_threshold:
                rule_subsets[1].append(rule)
                update_bins(rule, src_bins, dst_bins, bin_size)
            elif src_ip_range > self.ip_threshold and \
                    dst_ip_range <= self.ip_threshold:
                rule_subsets[2].append(rule)
                update_bins(rule, src_bins, dst_bins, bin_size)
            else:
                rule_subsets[3].append(rule)
        print(datetime.datetime.now(), "primary separate completed")

        # add subset 0 to other subsets if it is too small
        if len(rule_subsets[0]) <= self.leaf_threshold:
            for rule in rule_subsets[0]:
                src_ip_range, dst_ip_range = compute_ip_ranges(rule)
                if src_ip_range < dst_ip_range:
                    rule_subsets[1].append(rule)
                else:
                    rule_subsets[2].append(rule)
            rule_subsets[0] = []
        print(datetime.datetime.now(),
              "merge big rules completed, start merge small rules:",
              len(rule_subsets[3]))

        # add subset 3 to subset 1 and subset 2
        smallrule_idx = 0
        for rule in rule_subsets[3]:
            src_sum = sum([1 for i in src_bins if i > self.leaf_threshold])
            dst_sum = sum([1 for i in dst_bins if i > self.leaf_threshold])
            if src_sum < dst_sum or \
                    (src_sum == dst_sum and \
                        len(rule_subsets[1]) <= len(rule_subsets[2])):
                rule_subsets[1].append(rule)
                for i in range(
                        int(rule.ranges[0] // bin_size),
                        int(math.ceil(rule.ranges[1] / bin_size))):
                    src_bins[i] += 1
            else:
                rule_subsets[2].append(rule)
                for i in range(
                        int(rule.ranges[2] // bin_size),
                        int(math.ceil(rule.ranges[3] / bin_size))):
                    dst_bins[i] += 1

            smallrule_idx += 1
            if smallrule_idx % 100 == 0:
                print(datetime.datetime.now(), "merge small rules idx:",
                      smallrule_idx, " in ", len(rule_subsets[3]))

        print(datetime.datetime.now(), "merge small rules completed")
        # sort rule by priority
        for rule_subset in rule_subsets:
            rule_subset.sort(key=lambda i: i.priority)

        return rule_subsets[0:3]

    def select_action_hypersplit(self, tree, node):
        # select a dimension
        cut_dimension = -1
        cut_position = -1
        min_average = len(node.rules) * 2
        for i in range(5):
            # get distinct points
            left_points = set()
            right_points = set()
            all_points = set()
            for rule in node.rules:
                left = max(rule.ranges[i * 2], node.ranges[i * 2])
                right = min(rule.ranges[i * 2 + 1], node.ranges[i * 2 + 1]) - 1
                left_points.add(left)
                right_points.add(right)
                all_points.add(left)
                all_points.add(right)

            # expand distinct points to regions
            all_points = list(all_points)
            all_points.sort()
            region_points = []
            for point in all_points:
                if point in left_points:
                    region_points.append((point, 0))
                if point in right_points:
                    region_points.append((point, 1))

            if len(region_points) >= 3:
                # compute average rules in each region
                covered_rule_num = [0 for j in range(len(region_points) - 1)]
                for j in range(len(region_points) - 1):
                    for rule in node.rules:
                        if rule.ranges[i*2] <= region_points[j][0] and \
                                rule.ranges[i*2+1] > region_points[j+1][0]:
                            covered_rule_num[j] += 1
                average_covered_rule_num = sum(covered_rule_num) / (
                    len(region_points) - 1)

                # pick the dimension with the min average to cut
                if min_average > average_covered_rule_num:
                    min_average = average_covered_rule_num
                    cut_dimension = i

                    # compute the position to cut
                    half_covered_rule_num = sum(covered_rule_num) / 2
                    current_sum = covered_rule_num[0]
                    for i in range(1, len(region_points) - 1):
                        if region_points[i][1] == 0:
                            cut_position = region_points[i][0]
                        else:
                            cut_position = region_points[i][0] + 1

                        if current_sum > half_covered_rule_num:
                            break
                        current_sum += covered_rule_num[i]

        if cut_dimension == -1:
            print("cannot cut")

        return (cut_dimension, cut_position)

    def select_action_ficut(self, tree, node, cut_dimension):
        # compute the number of cuts
        range_left = node.ranges[cut_dimension * 2]
        range_right = node.ranges[cut_dimension * 2 + 1]

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

    def build_tree(self, rules, cut_algorithm, cut_dimension):
        tree = Tree(
            rules, self.leaf_threshold, {
                "node_merging": True,
                "rule_overlay": True,
                "region_compaction": False,
                "rule_pushup": False,
                "equi_dense": False
            })
        node = tree.get_current_node()
        count = 0
        while not tree.is_finish():
            if tree.is_leaf(node):
                node = tree.get_next_node()
                continue

            if cut_algorithm == "ficut":
                cut_dimension, cut_num = self.select_action_ficut(
                    tree, node, cut_dimension)

                # switch to hypersplit if the cut num is small
                if cut_num < self.ficut_threshold:
                    cut_algorithm = "hypersplit"
                else:
                    tree.cut_current_node(cut_dimension, cut_num)
                    node = tree.get_current_node()
            else:
                cut_dimension, cut_position = self.select_action_hypersplit(
                    tree, node)
                tree.cut_current_node_split(cut_dimension, cut_position)
                node = tree.get_current_node()

            count += 1
            if count % 10000 == 0:
                print(datetime.datetime.now(), "Depth:", tree.get_depth(),
                      "Remaining nodes:", len(tree.nodes_to_cut))
        return tree.compute_result()

    def train(self):
        print(datetime.datetime.now(), "Algorithm CutSplit")

        rule_subsets = self.separate_rules(self.rules)
        print(datetime.datetime.now(), "Separate rules completed")

        result = {"memory_access": 0, "bytes_per_rule": 0, "num_node": 0}
        for i, rule_subset in enumerate(rule_subsets):
            if len(rule_subset) == 0:
                continue

            cut_algorithm = "hypersplit" if i == 0 else "ficuts"
            cut_dimension = 0 if i == 1 else 1
            result_subset = self.build_tree(rule_subset, cut_algorithm,
                                            cut_dimension)
            result["memory_access"] += result_subset["memory_access"]
            result["bytes_per_rule"] += result_subset["bytes_per_rule"] * len(
                rule_subset)
            result["num_node"] += result_subset["num_node"]
        result["bytes_per_rule"] /= len(self.rules)

        print("%s Result %d %f %d" %
              (datetime.datetime.now(), result["memory_access"],
               result["bytes_per_rule"], result["num_node"]))
