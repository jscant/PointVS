"""
Use the output of CD-HIT sequence similarity clustering to split a dataset
into train and test, ensuring that no two proteins within the specified sequence
similarity are in the same split.
"""

import argparse
import random
from collections import defaultdict, deque, namedtuple
from pathlib import Path


def bfs(g, s):
    """Breadth first search algorithm.

    This will find all nodes connected to the source node s, the subgraph of
    all nodes connected to s in the parent graph g.

    Arguments:
        g: dictionary of node: neighbours, where neighbours is some iterable
            of nodes
        s: source node

    Returns:
        Set of all nodes in the same connected subgraph as the source.
    """
    visited = {s}
    queue = deque(g[s])
    while len(queue):
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue += g[node]
    return visited


def cdhit_output_to_graph(fname):
    """Generate sequence similarity graph from CD-HIT output."""
    g = defaultdict(deque)
    with open(Path(fname).expanduser(), 'r') as f:
        cluster = set()
        for line in f.readlines():
            if line.startswith('>Cluster'):
                for s in cluster:
                    g[s] += list(cluster.difference({s}))
                cluster.clear()
            else:
                pdbid = line.split('>')[-1].split('_')[0]
                cluster.add(pdbid)
    for key in g.keys():
        g[key] = deque(set(g[key]))
    return g


def generate_split(g, training_frac):
    """Randomly split data, ensuring no two PDBIDs with above the given
    sequence similarity threshold are in the same split."""
    train = set(g.keys())
    total_targets = len(train)
    val = set()
    while len(val) / total_targets < 1 - training_frac:
        source = random.sample(tuple(train), 1)[0]
        neighbours = bfs(g, source)
        train.remove(source)
        train -= neighbours
        val.add(source)
        val.update(neighbours)
    dataset = namedtuple('dataset', ['train', 'val'])
    return dataset(train, val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cdhit_output', type=str,
                        help='Clustering output of CD-HIT; usually named '
                             'xxx.out.clstr')
    parser.add_argument('train_frac', type=float,
                        help='Fraction in (0, 1) of dataset to be the training '
                             'set. The rest will be used as a test set.')
    args = parser.parse_args()

    g = cdhit_output_to_graph(args.cdhit_output)
    dataset = generate_split(g, args.train_frac)
    fname_base = Path(args.cdhit_output).name.split('.')[0]

    with open(fname_base + '.train', 'w') as f:
        f.write('\n'.join(dataset.train))
    with open(fname_base + '.test', 'w') as f:
        f.write('\n'.join(dataset.val))
