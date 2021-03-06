#!/usr/bin/env python3

# Build a single header out of a series of given files.
# Files are scanned for non-system includes, and everything is packed into the destination.

# TODO: Clean up include detection

import os
import networkx as nx
import re
import glob
import argparse

parser = argparse.ArgumentParser(description='Pack includes into a single header')

parser.add_argument('sources', metavar='I', nargs='+',
                    help = 'List of .h or .hpp files to pack')

parser.add_argument('destination',
                    help = 'destination file to write packed headers to')

args = parser.parse_args()

# print(args.sources)
# print(args.destination)

srcs = []

for i in args.sources:
    srcs += [ os.path.abspath(j) for j in glob.glob(i) ]

dest = args.destination

outf = open(dest, 'w')

G = nx.DiGraph()

def is_rel_include(line):
    if not line.strip().startswith("#"): return None
    ret = re.findall('^\s*\#include\s+["]([^">]+)*["]', line)
    if len(ret): return ret[0]
    return None

for f in srcs:
    container = os.path.dirname(f)
    # probe
    print("Considering:",f)
    deps = [ l for l in open(f) if l.strip().startswith("#include") ]
    deps = [ l.replace("#include","").strip() for l in deps ]
    deps = [ l.replace("\"","") for l in deps if not l.startswith("<")]
    deps = [ os.path.abspath(os.path.join(container, l)) for l in deps ]
    print("Depends on: ")
    for l in deps:
        G.add_edge(l,f)
        print(" -", l)
    print("")

print("Writing single header:")

for f in nx.topological_sort(G):
    print(" - Adding", f)
    for l in open(f):
        is_inc = is_rel_include(l)
        if not is_inc:
            outf.write(l)

print("Written to: {}".format(dest))
