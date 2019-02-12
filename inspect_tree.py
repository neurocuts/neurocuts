#!/usr/bin/env python
# Use this to visualize and check the correctness of pickled tree files.

import argparse
import pickle
import random

parser = argparse.ArgumentParser()

parser.add_argument(
    "file", type=str, help="The tree pkl file to load and analyze.")


def print_info(tree):
    print("This tree has {} rules".format(len(tree.rules)))
    print("Tree stats: {}".format(tree.compute_result()))
    print("Plottable visualization:\n{}".format(tree.stats_str()))


def check_classification(tree):
    failures = 0
    for i in range(10000):
        if i % 100 == 0:
            print("Testing randomly sampled packets", i)
        if random.random() > 0.5:
            packet = random.choice(tree.rules).sample_packet()
        else:
            packet = (random.randint(0, 2**32 - 1), random.randint(
                0, 2**32 - 1), random.randint(0, 2**16 - 1),
                      random.randint(0, 2**16 - 1), random.randint(
                          0, 2**5 - 1))
        expected_match = None
        for r in tree.rules:
            if r.matches(packet):
                expected_match = r
                break
        actual_match = tree.match(packet)
        expected_match = expected_match and tree.rules.index(expected_match)
        actual_match = actual_match and tree.rules.index(actual_match)
        if expected_match != actual_match:
            print("actual", actual_match, "expected", expected_match)
            failures += 1
    assert failures == 0, failures


def check_invariants(node):
    if node.children:
        if not node.is_partition():
            _check_disjointness(node.children)
        _check_rule_distribution(node)
        for c in node.children:
            check_invariants(c)
    else:
        if len(node.rules) > 16:
            print("WARNING: leaf node found with {} rules".format(
                len(node.rules)))


def _check_rule_distribution(node):
    for r in node.pruned_rules():
        count = 0
        for n in node.children:
            if r in n.rules:
                assert n.is_intersect_multi_dimension(r.ranges)
                count += 1
        if count == 0:
            assert False, ("Rule not found in any children", node.id, r.ranges,
                           [n.ranges for n in node.children])


def _check_disjointness(nodes):
    for ni in nodes:
        for nj in nodes:
            if ni != nj:
                assert not ni.is_intersect_multi_dimension(nj.ranges), \
                    (ni.ranges, nj.ranges)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.file, "rb") as f:
        tree = pickle.load(f)
    print_info(tree)
    check_invariants(tree.root)
    check_classification(tree)
    print("All checks ok, this looks like a valid tree.")
