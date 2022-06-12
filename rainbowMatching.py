import networkx as nx
import gurobipy as gp
import itertools
import sys
import time



######################################
# MODEL WITH FIXED BIPARTITE GRAPHS
######################################

def rainbow_matching_fixed(n, k, d, L):
    """
    Tries to find a counter example to: there exists a rainbow matching for bipartite F_1,...,F_k on vertex set [n] with
        each maximum degree d and |E(F_i)| > (k-1)d, where L_i is the left partition of F_i
    :param n: number of nodes in the graphs
    :param k: number of bipartite graphs
    :param d: maximum degree
    :param L: left partitions
    """
    # compute right-partitions given left-partitions
    R = [set(range(n)) - L[i] for i in range(len(L))]

    m = gp.Model("model")
    m.setParam("OutputFlag", 0)

    # variables for edges
    x = [m.addVars(
        [(l, r) for l, r in itertools.product(L[i], R[i])],
        vtype=gp.GRB.BINARY,
        name="x"
    ) for i in range(k)]

    # degree is at most d
    for i in range(k):
        for v in range(n):
            if v in L[i]:
                m.addConstr(
                    gp.quicksum(x[i][v, r] for r in R[i]) <= d
                )
            else:
                m.addConstr(
                    gp.quicksum(x[i][l, v] for l in L[i]) <= d
                )

    # size constraint on 1 <= i < k
    m.addConstrs(
        gp.quicksum(x[i][l, r] for (l, r) in itertools.product(L[i], R[i])) >= (k - 1) * d + 1
        for i in range(1, k)
    )

    # objective (size ``constraint'' on i = 0)
    m.setObjective(
        gp.quicksum(x[0][l, r] for (l, r) in itertools.product(L[0], R[0])),
        sense=gp.GRB.MAXIMIZE
    )

    # find all k-tuples
    Elist = [[] for i in range(k)]
    for i in range(k):
        for l, r in itertools.product(L[i], R[i]):
            Elist[i].append((l, r))

    # no rainbow graph (prohibit selecting k-disjoint tuple)
    m.addConstrs(
        gp.quicksum(x[i][t[i]] for i in range(k)) <= k - 1
        for t in itertools.product(*Elist) if len(set(sum(t,()))) == 2*k # 2k unique endpoints
    )

    print("Start optimizing fixed model")
    tstart = time.time()
    m.optimize()
    print("Finished optimizing in", time.time() - tstart, "seconds")

    if m.getAttr("Status") != 2:
        print("Model has no solution\n")
        return

    # report objective value
    objVal = m.getAttr("ObjVal")
    if objVal > (k - 1) * d:
        print("Found objective value", objVal, "> (k - 1) * d")
    else:
        print("Found objective value", objVal, "<= (k - 1) * d")

    # find edges of each graph
    F = [[t for t in itertools.product(L[i], R[i]) if x[i][t].X > 0.5] for i in range(k)]

    # check if correct solution
    correct = True
    for i in range(k):
        G = nx.empty_graph(n)
        G.add_edges_from(F[i])
        correct &= max([G.degree(j) for j in range(n)]) <= d
        correct &= len(F[i]) > (k - 1) * d

    # print assignment
    print("Found", correct, "solution")
    for i in range(k):
        print("F", i, "= (", L[i], "U", R[i], ",", F[i], ")")

    print("")

######################################
# MODEL CREATING BIPARTATE GRAPHS
######################################

def rainbow_matching_free(n, k, d):
    """
    Tries to find a counter example to: there exists a rainbow matching for bipartite F_1,...,F_k on vertex set [n] with
        each maximum degree d and |E(F_i)| > (k-1)d
    Does not require bipartite graphs to be constructed, included in ILP
    :param n: number of nodes in the graphs
    :param k: number of bipartite graphs
    :param d: maximum degree
    """
    M = 2 # large constant for OR-switching on bipartite property(>= 2)

    m = gp.Model("model")
    m.setParam("OutputFlag", 0)

    # variables for partition
    y = m.addVars(
        itertools.product(range(k), range(n)),
        vtype=gp.GRB.BINARY,
        name="y"
    )

    # variables for edges
    x = m.addVars(
        itertools.product(range(k), itertools.combinations(range(n), 2)),
        vtype=gp.GRB.BINARY,
        name="x"
    )

    # indicator variables for bipartite inequalities
    z = m.addVars(
        itertools.product(range(k), itertools.combinations(range(n), 2)),
        vtype=gp.GRB.BINARY,
        name="z"
    )

    # require bipartite: for fixed i: x_lr only 1 when y_l != y_r
    m.addConstrs(
        y[i, l] - y[i, r] >= x[i, (l, r)] - (z[i, (l, r)] * M)
        for i, (l, r) in itertools.product(range(k), itertools.combinations(range(n), 2))
    )
    m.addConstrs(
        y[i, l] - y[i, r] <= -x[i, (l, r)] + ((1 - z[i, (l, r)]) * M)
        for i, (l, r) in itertools.product(range(k), itertools.combinations(range(n), 2))
    )

    # degree is at most d
    m.addConstrs(
        gp.quicksum(x[i, tuple(sorted([v, u]))] for u in range(n) if u != v) <= d
        for i, v in itertools.product(range(k), range(n))
    )

    # size constraint on 1 <= i < k
    m.addConstrs(
        gp.quicksum(x[i, e] for e in itertools.combinations(range(n), 2)) >= (k - 1) * d + 1
        for i in range(1, k)
    )

    # objective (size ``constraint'' on i = 0)
    m.setObjective(
        gp.quicksum(x[0, e] for e in itertools.combinations(range(n), 2)),
        sense=gp.GRB.MAXIMIZE
    )

    # no rainbow graph (prohibit selecting k-disjoint tuple)
    m.addConstrs(
        gp.quicksum(x[i, t[i]] for i in range(k)) <= k - 1
        for t in itertools.product(itertools.combinations(range(n), 2), repeat=k) if len(set(sum(t,()))) == 2*k # 2k unique endpoints
    )

    print("Start optimizing free model")
    tstart = time.time()
    m.optimize()
    print("Finished optimizing in", time.time() - tstart, "seconds")

    # check if model has solution
    if m.getAttr("Status") != 2:
        print("Model has no solution\n")
        return

    # report objective value
    objVal = m.getAttr("ObjVal")
    if objVal > (k - 1) * d:
        print("Found objective value", objVal, "> (k - 1) * d")
    else:
        print("Found objective value", objVal, "<= (k - 1) * d")

    # get graph partitions and edges
    L = [set([v for v in range(n) if y[i, v].X <= 0.5]) for i in range(k)]
    R = [set([v for v in range(n) if y[i, v].X > 0.5]) for i in range(k)]
    F = [[e for e in itertools.combinations(range(n), 2) if x[i, e].X > 0.5] for i in range(k)]

    # check if correct solution
    correct = True
    for i in range(k):
        G = nx.empty_graph(n)
        G.add_edges_from(F[i])
        correct &= len(L[i].union(R[i])) == n
        for e in F[i]:
            correct &= (e[0] in L[i] and e[1] in R[i]) or (e[1] in L[i] and e[0] in R[i])
        correct &= max([G.degree(j) for j in range(n)]) <= d
        correct &= len(F[i]) > (k - 1) * d

    # print assignment
    print("Found", correct, "solution")
    for i in range(k):
        print("F", i, "= (", L[i], "U", R[i], ",", F[i], ")")



def main():
    n = 6 # number of nodes
    k = 3 # number of bipartite graphs
    d = 2 # max degree
    L = [{0,1,2},
         {1,2,3},
         {2,3,4},
        ] # left-partitions of the k bipartite graphs

    use_mode = 1 # which approach to take (0->fixed model, 1->free model, 2->both models)
    # NOTE: for the fixed model always the default (above) values for n, k, d, and L are used

    # read input
    if len(sys.argv) < 2:
        print("Using n=6, k=3, d=2 and running both models")

    elif len(sys.argv) >= 2:
        if sys.argv[1] == "fixed":
            use_mode = 0
        elif sys.argv[1] == "free":
            use_mode = 1
        elif sys.argv[1] == "both":
            use_mode = 2
        else :
            print("Usage: rainbowMatching.py <fixed|free|both> <n k d>")
            return

    if len(sys.argv) > 2:
        if len(sys.argv) != 5:
            print("Usage: rainbowMatching.py <fixed|free|both> <n k d>")
            return
        else:
            try:
                n = int(sys.argv[2])
                k = int(sys.argv[3])
                d = int(sys.argv[4])
            except:
                print("Please specify integers n, k, and d")
                print("Usage: rainbowMatching.py <fixed|free|both> <n k d>")
                return

    print("Running in mode", use_mode, "with n =", n, "k =", k, "d =", d)

    if use_mode == 0 or use_mode == 2: rainbow_matching_fixed(6, 3, 2, L)
    if use_mode == 1 or use_mode == 2: rainbow_matching_free(n, k, d)



if __name__ == "__main__":
    main()