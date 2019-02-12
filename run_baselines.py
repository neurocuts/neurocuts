#!/usr/bin/env python

import sys, os, time, subprocess, random, datetime

from hicuts import *
from hypercuts import *
from efficuts import *
from cutsplit import *

seed_files = []
acl_seed_files = []
fw_seed_files = []
ipc_seed_files = []

for i in range(1, 6):
    seed_files.append("acl%d_seed" % i)
    acl_seed_files.append("acl%d_seed" % i)
for i in range(1, 6):
    seed_files.append("fw%d_seed" % i)
    fw_seed_files.append("fw%d_seed" % i)
for i in range(1, 3):
    seed_files.append("ipc%d_seed" % i)
    ipc_seed_files.append("ipc%d_seed" % i)


def exe_cmd(cmd):
    #print "\t", cmd
    subprocess.call(cmd, shell=True)


def sync():
    exe_cmd("rsync -r * netx3:~/xinjin/neurocuts")


def gen_rules():
    for i in seed_files:
        for j in [1000, 10000, 100000]:
            cmd = "./db_generator/db_generator -bc parameter_files/%s " % i + \
                "%d 2 0.5 -0.1 %s_%d" % (j, i, j)
            exe_cmd(cmd)


EXPRS = ["1k"]


def run_hicuts():
    for i in EXPRS:
        print(i)
        rules = load_rules_from_file("classbench/acl1_%s" % i)
        cuts = HiCuts(rules)
        cuts.train()


def run_hypercuts():
    for i in EXPRS:
        print(i)
        rules = load_rules_from_file("classbench/acl1_%s" % i)
        cuts = HyperCuts(rules)
        cuts.train()


def run_efficuts():
    for i in EXPRS:
        print(i)
        rules = load_rules_from_file("classbench/acl1_%s" % i)
        cuts = EffiCuts(rules)
        cuts.train()


def run_cutsplit():
    for i in EXPRS:
        print(i)
        rules = load_rules_from_file("classbench/acl1_%s" % i)
        cuts = CutSplit(rules)
        cuts.train()


def run_all():
    # seed_files = ["ipc2_seed"]
    j_list = [1000, 10000, 100000]
    k_list = ["HyperCuts"]
    for i in seed_files:
        for j in j_list:
            # for j in [1000, 10000, 100000]:
            print("%s Rules %s_%d" % (datetime.datetime.now(), i, j))
            for k in k_list:  #, "CutSplit"]:
                rules = load_rules_from_file("classbench/%s_%d" % (i, j))
                cuts = None
                if k == "HiCuts":
                    cuts = HiCuts(rules)
                elif k == "HyperCuts":
                    cuts = HyperCuts(rules)
                elif k == "EffiCuts":
                    cuts = EffiCuts(rules)
                elif k == "CutSplit":
                    cuts = CutSplit(rules)
                cuts.train()


def run_all_hicuts(files_type):
    # seed_files = ["ipc2_seed"]
    if files_type == "acl":
        i_list = acl_seed_files
    elif files_type == "fw":
        i_list = fw_seed_files
    elif files_type == "ipc":
        i_list = ipc_seed_files
    else:
        i_list = seed_files
    i_list = ["acl3_seed"]
    j_list = [100000]
    k_list = ["HiCuts"]
    for j in j_list:
        for i in i_list:
            # for j in [1000, 10000, 100000]:
            print("%s Rules %s_%d" % (datetime.datetime.now(), i, j))
            for k in k_list:  #, "CutSplit"]:
                rules = load_rules_from_file("classbench/%s_%d" % (i, j))
                cuts = None
                if k == "HiCuts":
                    cuts = HiCuts(rules)
                elif k == "HyperCuts":
                    cuts = HyperCuts(rules)
                elif k == "EffiCuts":
                    cuts = EffiCuts(rules)
                elif k == "CutSplit":
                    cuts = CutSplit(rules)
                cuts.train()


def run_all_hypercuts(files_type):
    # seed_files = ["ipc2_seed"]
    if files_type == "acl":
        i_list = acl_seed_files
    elif files_type == "fw":
        i_list = fw_seed_files
    elif files_type == "ipc":
        i_list = ipc_seed_files
    elif files_type == "all":
        i_list = seed_files
    # i_list = [files_type]
    # j_list = [1000, 10000, 100000]
    j_list = [100000]
    k_list = ["HyperCuts"]
    for i in i_list:
        for j in j_list:
            # for j in [1000, 10000, 100000]:
            print("%s Rules %s_%d" % (datetime.datetime.now(), i, j))
            for k in k_list:  #, "CutSplit"]:
                rules = load_rules_from_file("classbench/%s_%d" % (i, j))
                cuts = None
                if k == "HiCuts":
                    cuts = HiCuts(rules)
                elif k == "HyperCuts":
                    cuts = HyperCuts(rules)
                elif k == "EffiCuts":
                    cuts = EffiCuts(rules)
                elif k == "CutSplit":
                    cuts = CutSplit(rules)
                cuts.train()


def run_file(file_name, alg):
    k = alg
    rules = load_rules_from_file("classbench/%s" % (file_name))
    cuts = None
    if k == "HiCuts":
        cuts = HiCuts(rules)
    elif k == "HyperCuts":
        cuts = HyperCuts(rules)
    elif k == "EffiCuts":
        cuts = EffiCuts(rules)
    elif k == "CutSplit":
        cuts = CutSplit(rules)
    cuts.train()


def run_all_efficuts(files_type):
    # seed_files = ["ipc2_seed"]
    if files_type == "acl":
        i_list = acl_seed_files
    elif files_type == "fw":
        i_list = fw_seed_files
    elif files_type == "ipc":
        i_list = ipc_seed_files
    else:
        i_list = seed_files
    j_list = [1000, 10000, 100000]
    # j_list = [10000]
    k_list = ["EffiCuts"]
    for j in j_list:
        for i in i_list:
            # for j in [1000, 10000, 100000]:
            print("%s Rules %s_%d" % (datetime.datetime.now(), i, j))
            for k in k_list:  #, "CutSplit"]:
                rules = load_rules_from_file("classbench/%s_%d" % (i, j))
                cuts = None
                if k == "HiCuts":
                    cuts = HiCuts(rules)
                elif k == "HyperCuts":
                    cuts = HyperCuts(rules)
                elif k == "EffiCuts":
                    cuts = EffiCuts(rules)
                elif k == "CutSplit":
                    cuts = CutSplit(rules)
                cuts.train()


def gen_result(file_name):
    fin = open(file_name)
    rules = ""
    algorithm = ""
    oneline = fin.readline()
    while oneline != "":
        items = oneline.strip().split()
        if len(items) == 4 and items[2] == "Rules":
            rules = items[3]
        elif len(items) == 4 and items[2] == "Algorithm":
            algorithm = items[3]
        elif len(items) == 5 and items[2] == "Result":
            print("%s %s %s %s" % (rules, algorithm, items[3], items[4]))
        oneline = fin.readline()
    fin.close()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage")
        print("  tool.py sync")
        print("  tool.py gen_rules")
        print("  tool.py expr")
        sys.exit()

    if sys.argv[1] == "sync":
        sync()
    elif sys.argv[1] == "gen_rules":
        gen_rules()
    elif sys.argv[1] == "hicuts":
        run_all_hicuts(sys.argv[2])
    elif sys.argv[1] == "hypercuts":
        run_all_hypercuts(sys.argv[2])
    elif sys.argv[1] == "efficuts":
        run_all_efficuts(sys.argv[2])
    elif sys.argv[1] == "run_file":
        run_file(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "expr":
        #        run_all()
        run_hicuts()
#        run_hypercuts()
#        run_efficuts()
#        run_cutsplit()
    elif sys.argv[1] == "gen_result":
        gen_result(sys.argv[2])
    else:
        print("Not supported option")
