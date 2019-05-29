import sys

from tree import *
from hicuts import *
from hypercuts import *
from efficuts import *
from cutsplit import *


def test_tree_():
    print("========== rule ==========")
    rule = Rule([0, 10, 0, 10, 10, 20, 0, 1, 0, 1])
    print(rule)
    print("True", rule.is_intersect(2, 0, 11))
    print("False", rule.is_intersect(2, 0, 10))
    print("False", rule.is_intersect(2, 20, 21))
    print("True",
          rule.is_intersect_multi_dimension([0, 10, 0, 10, 0, 11, 0, 1, 0, 1]))
    print("False",
          rule.is_intersect_multi_dimension([0, 10, 0, 10, 0, 10, 0, 1, 0, 1]))
    print(
        "False",
        rule.is_intersect_multi_dimension([0, 10, 0, 10, 20, 21, 0, 1, 0, 1]))

    print("========== node ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 10, 20, 10, 15, 0, 0]))
    rules.append(Rule([0, 100, 0, 100, 0, 100, 20, 30, 0, 0]))
    rules.append(Rule([0, 100, 0, 100, 0, 100, 40, 50, 0, 0]))
    ranges = [0, 1000, 0, 1000, 0, 1000, 0, 1000, 0, 1000]
    node = Node(0, ranges, rules, 1)
    print(node)

    print("========== tree single-dimensional cuts ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    tree = Tree(rules, 1)
    tree.refinement_region_compaction(tree.root)
    print(tree.root)
    tree.cut_current_node(0, 2)
    tree.print_layers()

    tree.cut_current_node(1, 2)
    tree.get_next_node()
    tree.get_next_node()
    tree.cut_current_node(1, 2)
    tree.print_layers()

    print("========== tree multi-dimensional cuts ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    tree = Tree(rules, 1)
    tree.refinement_region_compaction(tree.root)
    tree.cut_current_node_multi_dimension([0, 1, 2, 3, 4], [2, 2, 1, 1, 1])
    tree.print_layers()

    print("========== print tree ==========")
    print(tree)

    print("========== load rule ==========")
    rules = load_rules_from_file("classbench/acl1_20")
    for rule in rules:
        print(rule)

    # check continuous region
    node1 = Node(0, [0, 10, 0, 10, 10, 20, 10, 15, 0, 10], None, 1)
    node2 = Node(0, [0, 10, 10, 20, 10, 20, 10, 15, 0, 10], None, 1)
    node3 = Node(0, [10, 20, 20, 30, 10, 20, 10, 15, 0, 10], None, 1)
    node4 = Node(0, [20, 30, 20, 30, 10, 20, 10, 15, 0, 10], None, 1)
    assert tree.check_contiguous_region(node1, node2)
    assert not tree.check_contiguous_region(node2, node3)
    assert tree.check_contiguous_region(node3, node4)


def test_tree():
    tree = Tree([], 1)

    # check continuous region
    node0 = Node(0, [0, 10, 0, 10, 10, 20, 10, 15, 0, 10], None, 1)
    node1 = Node(0, [0, 10, 10, 20, 10, 20, 10, 15, 0, 10], None, 1)
    node2 = Node(0, [10, 20, 20, 30, 10, 20, 10, 15, 0, 10], None, 1)
    node3 = Node(0, [20, 30, 20, 30, 10, 20, 10, 15, 0, 10], None, 1)
    assert tree.check_contiguous_region(node0, node1)
    assert not tree.check_contiguous_region(node1, node2)
    assert tree.check_contiguous_region(node2, node3)

    # refinement equi dense
    rule0 = Rule(0, [0, 10, 0, 10, 10, 20, 0, 1, 0, 1])
    rule1 = Rule(1, [0, 10, 0, 10, 10, 20, 0, 1, 0, 2])
    rule2 = Rule(2, [0, 10, 0, 10, 10, 20, 0, 1, 0, 3])
    rule3 = Rule(3, [0, 10, 0, 10, 10, 20, 0, 1, 0, 4])
    node0 = Node(0, [0, 10, 0, 10, 10, 20, 0, 1, 0, 1], [rule0], 1)
    node1 = Node(1, [0, 10, 10, 20, 10, 20, 0, 1, 0, 1], [rule0, rule1], 1)
    node2 = Node(2, [10, 20, 0, 10, 10, 20, 0, 1, 0, 1], [rule1], 1)
    node3 = Node(3, [10, 20, 10, 20, 10, 20, 0, 1, 0, 1], [rule2], 1)
    node4 = Node(4, [20, 30, 20, 30, 10, 20, 0, 1, 0, 1],
                 [rule0, rule1, rule2], 1)
    nodes = [node0, node1, node2, node3, node4]
    nodes = tree.refinement_equi_dense(nodes)
    assert len(nodes) == 4
    assert nodes[0].id == node0.id
    assert nodes[0].ranges == [0, 10, 0, 20, 10, 20, 0, 1, 0, 1]
    assert nodes[0].rules == [rule0, rule1]
    assert nodes[0].depth == node0.depth
    assert nodes[1].id == node2.id
    assert nodes[1].ranges == node2.ranges
    assert nodes[1].rules == node2.rules
    assert nodes[1].depth == node2.depth
    assert nodes[2].id == node3.id
    assert nodes[3].id == node4.id

    node0 = Node(0, [0, 10, 0, 10, 10, 20, 0, 1, 0, 1], [rule0], 1)
    node1 = Node(1, [0, 10, 10, 20, 10, 20, 0, 1, 0, 1], [rule0, rule1], 1)
    node2 = Node(2, [10, 20, 0, 10, 10, 20, 0, 1, 0, 1], [rule0], 1)
    node3 = Node(3, [10, 20, 10, 20, 10, 20, 0, 1, 0, 1], [rule0, rule1], 1)
    node4 = Node(4, [20, 30, 20, 30, 10, 20, 0, 1, 0, 1],
                 [rule0, rule1, rule2], 1)
    nodes = [node0, node1, node2, node3, node4]
    nodes = tree.refinement_equi_dense(nodes)
    assert len(nodes) == 2
    assert nodes[0].id == node0.id
    assert nodes[0].ranges == [0, 20, 0, 20, 10, 20, 0, 1, 0, 1]
    assert nodes[0].rules == [rule0, rule1]
    assert nodes[0].depth == node0.depth
    assert nodes[1].id == node4.id
    assert nodes[1].ranges == node4.ranges
    assert nodes[1].rules == node4.rules
    assert nodes[1].depth == node4.depth


def test_refinements():
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    tree = Tree(rules, 1)

    print("========== node merging ==========")
    rules1 = []
    rules1.append(Rule([0, 10, 0, 10, 10, 20, 10, 15, 0, 1]))
    rules1.append(Rule([0, 100, 0, 100, 0, 100, 20, 30, 0, 1]))
    rules1.append(Rule([0, 100, 0, 100, 0, 100, 40, 50, 0, 1]))
    rules2 = [rule for rule in rules1]
    rules3 = [rule for rule in rules1]
    ranges = [0, 1000, 0, 1000, 0, 1000, 0, 1000, 0, 1000]
    node1 = Node(0, [0, 100, 0, 100, 0, 1000, 0, 1000, 0, 1000], rules1, 1)
    node2 = Node(1, [0, 100, 100, 200, 0, 1000, 0, 1000, 0, 1000], rules2, 1)
    node3 = Node(1, [0, 100, 100, 200, 0, 1000, 0, 1000, 0, 1000], rules3[1:],
                 1)
    print("True", tree.refinement_node_merging(node1, node2))
    print("False", tree.refinement_node_merging(node1, node3))

    node1 = Node(0, [0, 100, 0, 100, 0, 1000, 0, 1000, 0, 1000], rules1, 1)
    node2 = Node(1, [0, 100, 100, 200, 0, 1000, 0, 1000, 0, 1000], rules2, 1)
    node3 = Node(1, [0, 100, 100, 200, 0, 1000, 0, 1000, 0, 1000], rules3, 1)
    node4 = Node(1, [0, 100, 200, 300, 0, 1000, 0, 1000, 0, 1000], rules3, 1)
    tree.update_tree(tree.root, [node1, node2, node3, node4])
    print(node1)
    print(node3)

    print("========== rule overlay ==========")
    rule1 = Rule([0, 10, 0, 10, 10, 20, 10, 15, 0, 1])
    rule2 = Rule([0, 10, 0, 10, 10, 20, 10, 15, 0, 1])
    rule3 = Rule([0, 12, 0, 10, 10, 20, 10, 15, 0, 1])
    rule4 = Rule([0, 9, 0, 10, 10, 20, 10, 15, 0, 1])
    rule5 = Rule([0, 9, 0, 10, 10, 20, 10, 15, 0, 2])
    ranges = [0, 10, 0, 10, 10, 20, 10, 15, 0, 2]
    print("True", rule1.is_covered_by(rule2, ranges))
    print("True", rule1.is_covered_by(rule3, ranges))
    print("False", rule1.is_covered_by(rule4, ranges))

    node1 = Node(0, ranges, [rule1, rule2, rule3, rule4, rule5], 1)
    tree.refinement_rule_overlay(node1)
    print(node1)

    ranges = [0, 9, 0, 10, 10, 20, 10, 15, 0, 1]
    print("True", rule1.is_covered_by(rule4, ranges))

    node1 = Node(0, ranges, [rule1, rule2, rule3, rule4, rule5], 1)
    tree.refinement_rule_overlay(node1)
    print(node1)

    print("========== region compaction ==========")
    rules1 = []
    rules1.append(Rule([0, 10, 0, 10, 10, 20, 10, 15, 0, 1]))
    rules1.append(Rule([0, 100, 0, 100, 0, 100, 20, 30, 0, 1]))
    rules1.append(Rule([0, 100, 0, 100, 0, 100, 40, 50, 0, 1]))
    ranges = [0, 1000, 0, 1000, 0, 1000, 0, 1000, 0, 1000]
    node1 = Node(0, ranges, rules1, 1)
    print(node1)
    tree.refinement_region_compaction(node1)
    print(node1)

    print("========== rule pushup ==========")
    rule1 = Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1])
    rule2 = Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1])
    rule3 = Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1])
    rule4 = Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1])
    ranges = [0, 1000, 0, 1000, 0, 1000, 0, 1000, 0, 1000]
    tree = Tree([rule1, rule2, rule3, rule4], 1)
    node1 = tree.create_node(1, ranges.copy(), [rule1, rule2, rule3], 2)
    node2 = tree.create_node(2, ranges.copy(), [rule2, rule4], 2)
    tree.update_tree(tree.root, [node1, node2])
    node3 = tree.create_node(3, ranges.copy(), [rule1, rule2, rule3], 3)
    node4 = tree.create_node(4, ranges.copy(), [rule1, rule2], 3)
    tree.update_tree(node1, [node3, node4])
    tree.depth = 3
    tree.print_layers()

    tree.refinement_rule_pushup()
    tree.print_layers()


def test_hicuts():
    print("========== hicuts ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    cuts = HiCuts(rules)
    cuts.train()


def test_hypercuts():
    print("========== hypercuts ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    cuts = HyperCuts(rules)
    cuts.leaf_threshold = 1
    cuts.train()


def test_efficuts():
    print("========== efficuts ==========")

    # test separate rules
    rule0 = Rule([
        0, 2**32 * 0.05, 0, 2**32 * 0.05 - 1, 0, 2**16 * 0.5 - 1, 0,
        2**16 * 0.5, 0, 2**8 * 0.5 - 1
    ])
    rule1 = Rule([
        0, 2**32 * 0.05, 0, 2**32 * 0.05 - 1, 0, 2**16 * 0.5 - 1, 0,
        2**16 * 0.5, 0, 2**8 * 0.5
    ])
    rules = [rule0, rule1]
    cuts = EffiCuts(rules)
    rule_subsets = cuts.separate_rules(rules)
    assert rule_subsets[18] == [rule0]
    assert rule_subsets[19] == [rule1]

    # test merge rule subsets
    rule_subsets = [[] for i in range(32)]
    rule_subsets[0] = [0]
    rule_subsets[10] = [10]
    rule_subsets[24] = [24]
    rule_subsets[26] = [26]
    rule_subsets[27] = [27]
    rule_subsets[28] = [28]
    rule_subsets[29] = [29]
    rule_subsets[31] = [31]

    rule_subsets = cuts.merge_rule_subsets(rule_subsets)
    assert rule_subsets[0] == [26, 27]
    assert rule_subsets[1] == [28, 29]
    assert rule_subsets[2] == [0]
    assert rule_subsets[3] == [10]
    assert rule_subsets[4] == [24]
    assert rule_subsets[5] == [31]


def test_cutsplit():
    print("========== cutsplit ==========")

    # test separate rules
    rule0 = Rule(0, [0, 2**12, 0, 2**12, 0, 1, 0, 1, 0, 1])
    rule1 = Rule(1, [2**8, 2**24, 0, 2**25, 0, 1, 0, 1, 0, 1])
    rule2 = Rule(2, [0, 2**25, 2**20, 2**24, 0, 1, 0, 1, 0, 1])
    rule3 = Rule(3, [0, 2**32, 0, 2**32, 0, 1, 0, 1, 0, 1])
    cuts = CutSplit(None)
    cuts.leaf_threshold = 0
    rule_subsets = cuts.separate_rules([rule0, rule1, rule2, rule3])
    assert rule_subsets[0] == [rule0]
    assert rule_subsets[1] == [rule1]
    assert rule_subsets[2] == [rule2, rule3]

    cuts.leaf_threshold = 1
    rule_subsets = cuts.separate_rules([rule0, rule1, rule2, rule3])
    assert rule_subsets[0] == []
    assert rule_subsets[1] == [rule1, rule3]
    assert rule_subsets[2] == [rule0, rule2]

    rule2 = Rule(2, [0, 2**25, 2**8, 2**24, 0, 1, 0, 1, 0, 1])
    cuts.leaf_threshold = 0
    rule_subsets = cuts.separate_rules([rule0, rule1, rule2, rule3])
    assert rule_subsets[0] == [rule0]
    assert rule_subsets[1] == [rule1, rule3]
    assert rule_subsets[2] == [rule2]

    # test select action
    rule0 = Rule(0, [0, 2**12, 0, 2**12 + 1, 0, 1, 0, 1, 0, 1])
    rule1 = Rule(1, [2**8, 2**24, 2**12, 2**25, 0, 1, 0, 1, 0, 1])
    rule2 = Rule(2, [0, 2**25, 2**20, 2**24, 0, 1, 0, 1, 0, 1])
    rule3 = Rule(3, [0, 2**32, 0, 2**32, 0, 1, 0, 1, 0, 1])
    node = Node(0, [0, 2**32, 0, 2**32, 0, 2**16, 0, 2**16, 0, 2**8],
                [rule0, rule1, rule2, rule3], 1)
    cut_dimension, cut_position = cuts.select_action(None, node)
    assert cut_dimension == 1 and cut_position == 1048576


if __name__ == "__main__":
    #test_tree()
    #test_refinements()
    #test_hicuts()
    #test_hypercuts()
    #test_efficuts()
    test_cutsplit()
