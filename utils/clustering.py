import itertools
from collections import Counter
from sklearn.cluster import AgglomerativeClustering


class TokenSet:
    def __init__(self, tokens: list):
        self.tokens = tokens or []

    def __add__(self, other):
        return TokenSet(self.tokens + other.tokens)

    def common_words(self, n=5):
        return [token for token, _ in Counter(self.tokens).most_common(n)]

    def __repr__(self) -> str:
        return str(self.tokens)


class Node:
    def __init__(
        self,
        id,
        label=None,
        tokens=None,
        text: TokenSet = None,
        children_count=0,
        children=None
    ):
        self.id = id
        self.label = label if label is not None else 'merge'
        self.children = children or []
        self.tokens = tokens if tokens else TokenSet([])
        self.children_count = children_count
        self.text = text

    def clone(
        self,
        id=None,
        label=None,
        children=None,
        tokens=None,
        children_count=None,
        text=None,
    ):
        return Node(
            id=id if id is not None else self.id,
            label=label if label is not None else self.label,
            children=children if children is not None else self.children,
            tokens=tokens if tokens is not None else self.tokens,
            children_count=children_count if children_count is not None else self.children_count,
            text=text if text is not None else self.text,
        )

    def __repr__(self) -> str:
        return f"Node(id={self.id}, label={self.label}, tokens={self.tokens.common_words()}, count={self.children_count}, children={len(self.children)})"

    def add_child(self, child):
        self.children.append(child)

    @property
    def super_merge_node(self):
        labels = [child.label for child in self.children]
        return 'merge' in labels

    def header(self, father=None):
        if father is None:
            try:
                return self.tokens.common_words(1)[0]
            except:
                return None

        faather_word = father.tokens.common_words(1)
        words = self.tokens.common_words(2)
        if not words:
            return None

        if not faather_word or len(words) < 2 or words[0] != faather_word[0]:
            return words[0]

        return words[1]

    def words(self):
        if self.text:
            return self.text

        if self.tokens:
            return ' '.join(self.tokens.common_words())

        return None


def build_agglomerative_tree(cluster: AgglomerativeClustering, nodes: list[Node]):
    ii = itertools.count(cluster.labels_.shape[0])
    whole_tree = [
        {'node_id': next(ii), 'left': x[0], 'right': x[1]}
        for x in cluster.children_
    ]

    nodes = {
        index: nodes[index].clone(label=label)
        for index, label in enumerate(cluster.labels_)
    }

    for node in whole_tree:
        if node['node_id'] not in nodes:
            left = nodes[node['left']]
            right = nodes[node['right']]

            if left.label == right.label:
                instance = Node(node['node_id'], left.label)
            else:
                instance = Node(node['node_id'])

            nodes[instance.id] = instance
            instance.add_child(left)
            instance.add_child(right)

    root = nodes[node['node_id']]
    return root


####################################################################
#
#   Visitors
#
###################################################################

def count_visit(node: Node):
    children = [count_visit(child) for child in node.children]

    return node.clone(
        children=children,
        children_count=sum([child.children_count for child in children]) + 1
    )


def compute_tokens_visit(node: Node):
    if not node.children:
        return node

    children = [compute_tokens_visit(child) for child in node.children]
    new_node = node.clone(children=children, tokens=None)
    for child in children:
        new_node.tokens += child.tokens

    return new_node


def compress_clusters(node: Node):
    if node.label != 'merge':
        return node.clone(children=[])

    return node.clone(children=[
        compress_clusters(child) for child in node.children
    ])


def cut_clusters(node: Node, threshold=5):
    if not node.children:
        return node

    children = []

    for child in node.children:
        if child.children_count >= threshold and (
                new_child := cut_clusters(child, threshold)):
            children.append(new_child)

    if len(children) == 0:
        return None

    if len(children) == 1:
        return children[0]

    return node.clone(children=children)


def balance_visit(node: Node):
    children = node.children

    while children:
        children = sorted(
            children, key=lambda x: x.children_count, reverse=True
        )
        child = children.pop(0)

        if child.label == "merge":
            children.extend(child.children)
        else:
            children.append(child)
            break

    return node.clone(children=[balance_visit(child) for child in children])


def renumber(node: Node, index=1):
    return node.clone(id=index, children=[
        renumber(child, index + i + 1)
        for i, child in enumerate(node.children)
    ])


def filter_clusters(node: Node, max=float('inf'), min=float('-inf')):
    children = [child for child in node.children if min <
                child.children_count < max]
    return node.clone(children=children, tokens=[])


def _aux_build_list(node: Node, result=None, father=None, whole_tree=False):
    result = result or []
    for child in node.children:
        result += build_list(child, father, whole_tree)

    return result


def build_list(node: Node, father=None, whole_tree=False):
    if whole_tree or node.tokens:
        result = [{
            'id': node.id,
            'label': node.label,
            'parent': father if father is None else father.id,
            'text': node.header(father),
            'words': node.words(),
            'children_count': node.children_count,
        }]

        return _aux_build_list(node, result, node, whole_tree)

    else:
        return _aux_build_list(node, father=father)


####################################################################
#
#   Find Clusters
#
###################################################################

def compute_intersection(group):
    return set.intersection(*tuple(set(l) for l in group))


def build_cluster(matrix, dist=0.8):
    clustering = AgglomerativeClustering(
        metric='precomputed',  # Use the custom distance matrix
        linkage='complete',  # ont of 'ward', 'complete', 'average', 'single'},
        distance_threshold=dist,  # Adjust for automatic clustering
        n_clusters=None  # Auto-detect number of clusters
    )

    clustering.fit(matrix)
    return clustering


def find_clusters(df, matrix):
    dist = 0.1
    _clustering = None
    while dist < 1:
        print("Testing:", dist)
        clustering = build_cluster(matrix, dist)
        df['label'] = clustering.labels_

        df_grouped = df.groupby('label').agg({
            'name_tokens': compute_intersection,
            'name': 'count'
        }).reset_index()

        df_grouped['len'] = df_grouped['name_tokens'].apply(len)
        if len(df_grouped[df_grouped['len'] == 0]):
            break

        _clustering = clustering
        dist += 0.1

    return _clustering
