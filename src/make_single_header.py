import sys
import os
import networkx as nx
import re

srcs = [ os.path.abspath(i) for i in sys.argv[1:-1] ]
dest = sys.argv[-1]

print(srcs, dest)

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
    print("Considering: ",f)
    deps = [ l for l in open(f) if l.strip().startswith("#include") ]
    deps = [ l.replace("#include","").strip() for l in deps ]
    deps = [ l.replace("\"","") for l in deps if not l.startswith("<")]
    deps = [ os.path.abspath(os.path.join(container, l)) for l in deps ]
    print("Depends on: ")
    for l in deps:
        G.add_edge(l,f)
        print(" - ", l)

print("Writing single header:")

for f in nx.topological_sort(G):
    print("- Adding ", f)
    for l in open(f):
        is_inc = is_rel_include(l)
        if not is_inc:
            outf.write(l)
