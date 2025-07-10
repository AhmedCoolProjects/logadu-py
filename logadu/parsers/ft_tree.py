#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Frequent Pattern Tree (FP-Tree) implementation for log template extraction.

This module implements a tree-based approach to extract log templates from system logs.
It processes logs by:
1. Tokenizing log messages
2. Building frequency trees based on word occurrences
3. Extracting common patterns as templates

Key features:
- Handles different process IDs (PIDs)
- Supports pruning strategies
- Can visualize the generated trees
- Handles both template extraction and template matching modes
"""

import os
from typing import List, Dict, Tuple, Optional, Set
from copy import deepcopy
import click

# Global variable to track maximum number of children
# max_org = 0

class Node:
    """Node of the FP-Tree.

    Attributes:
        _data: The word stored in this node
        _children: List of child nodes
        _level: Depth level in the tree
        _index: Unique identifier for visualization
        _father: Parent node reference
        _no_cutting: Flag to prevent pruning (1 = no pruning)
        _change_to_leaf: Flag to mark node as leaf
        is_end_node: Marks if node is the end of a template
    """

    def __init__(self, data: str):
        self._data = data
        self._children = []
        self._level = 0
        self._index = 0
        self._father = ''
        self._no_cutting = 0
        self._change_to_leaf = 0
        self.is_end_node = 0

    def get_data(self) -> str:
        return self._data

    def get_children_num(self) -> int:
        return len(self._children)

    def get_children(self) -> List['Node']:
        return self._children

    def delete_children(self) -> None:
        """Mark this node as a leaf by removing all children."""
        self._change_to_leaf = 1
        self._children = []

    def add_child_node(self, node: 'Node', leaf_num: int = 10, cut_level: int = 3, rebuild: int = 0) -> bool:
        """Add a child node with pruning control.
        
        Args:
            node: Child node to add
            leaf_num: Maximum allowed children before pruning
            cut_level: Depth level where pruning starts
            rebuild: Flag indicating tree rebuild mode
            
        Returns:
            True if child was added, False otherwise
        """
        _max_org = 0
        
        node._level = self._level + 1
        
        # Update maximum observed children count
        if _max_org < len(self._children):
            _max_org = len(self._children)

        # Apply pruning rules
        if (self._level > cut_level and 
            len(self._children) == leaf_num and 
            self._no_cutting != 1 and 
            rebuild == 0):
            self.delete_children()
            return False

        if self._change_to_leaf == 1:
            return False

        node._father = self
        self._children.append(node)
        return True

    def find_child_node(self, data: str) -> Optional['Node']:
        """Find child node with matching data."""
        for child in self._children:
            if child.get_data() == data:
                return child
        return None


class Tree:
    """Template tree structure for storing log patterns.

    Attributes:
        _head: Root node of the tree
    """

    def __init__(self, head: str):
        self._head = Node(head)
        self._head._level = 1

    def link_to_head(self, node: Node, leaf_num: int = 10) -> None:
        """Set the root node of the tree."""
        self._head.add_child_node(node, leaf_num)

    def insert_node(self, path: List[str], data: str, para: Dict, 
                   is_end_node: int = 0, no_cutting: int = 0, 
                   rebuild: int = 0) -> bool:
        """Insert a node into the tree.
        
        Args:
            path: List of words leading to the parent node
            data: Word to insert
            para: Parameters dictionary
            is_end_node: Marks if this node completes a template
            no_cutting: Prevents pruning this branch
            rebuild: Tree rebuild mode flag
            
        Returns:
            True if node was inserted, False otherwise
        """
        NO_CUTTING = 0
        if rebuild == 0:
            NO_CUTTING = para['NO_CUTTING']
        leaf_num = para['leaf_num']

        cur = self._head
        for step in path:
            if cur._change_to_leaf == 1:
                return False
            child = cur.find_child_node(step)
            if not child:
                return False
            cur = child

        # Check if node already exists
        for child in cur.get_children():
            if child.get_data() == data:
                if child.is_end_node == 0:
                    child.is_end_node = is_end_node
                return False

        # Create and add new node
        new_node = Node(data)
        if rebuild == 1:
            new_node._no_cutting = 1
        elif no_cutting and NO_CUTTING:
            new_node._no_cutting = 1
        new_node.is_end_node = is_end_node
        cur.add_child_node(new_node, leaf_num, rebuild=rebuild)
        return True

    def search_path(self, path: List[str]) -> Optional[Node]:
        """Search for a node following the given path."""
        cur = self._head
        for step in path:
            cur = cur.find_child_node(step)
            if not cur:
                return None
        return cur


class WordsFrequencyTree:
    """Main class for building and processing frequency trees."""

    def __init__(self):
        self.tree_list = {}  # Dictionary of trees: {pid: Tree object}
        self.paths = []
        self._nodes = []

    def _init_tree(self, pids: List[str]) -> None:
        """Initialize trees for each PID."""
        self.tree_list = {}
        for pid in pids:
            self.tree_list[pid] = Tree(pid)

    def _traversal(self, subtree: Node, path: List[List[str]], sub_path: List[Tuple[str]]) -> None:
        """Recursive helper for tree traversal."""
        subs = subtree.get_children()

        if not subs:
            path.append(self._nodes)
            self._nodes = self._nodes[:-1]
            return None
        else:
            if subtree.is_end_node == 1:
                _path = tuple(deepcopy(self._nodes))
                sub_path.append(_path)
                subtree.is_end_node = 0

            for n in subs:
                self._nodes.append(n.get_data())
                self._traversal(n, path, sub_path)
            self._nodes = self._nodes[:-1]

    def traversal_tree(self, tree: Tree) -> List:
        """Traverse tree and extract all templates.
        
        Returns:
            List containing [root_pid, list_of_templates]
        """
        _nodes, path, sub_path = [], [], []
        path.append(tree._head.get_data())
        self._traversal(tree._head, path, sub_path)
        path.extend(sub_path)
        _path = [tuple(x) for x in path[1:]]
        return [path[0], list(set(_path))]

    def auto_temp(self, logs: List[Tuple[str, List[str]]], 
                 words_frequency: List[str], para: Dict, rebuild: int = 0) -> None:
        """Build the frequency tree from logs.
        
        Args:
            logs: List of (pid, words) tuples
            words_frequency: Ordered list of words by frequency
            para: Parameters dictionary
            rebuild: Flag for rebuild mode
        """
        leaf_num = para['leaf_num']
        CUTTING_PERCENT = para['CUTTING_PERCENT'] if rebuild == 0 else 0

        for log in logs:
            pid, words = log
            words = list(set(words))  # Remove duplicates
            
            # Create word index based on frequency
            words_index = {}
            words_count = {}
            for word in words:
                if word in words_frequency:
                    words_index[word] = float(words_frequency.index(word))
                words_count[word] = words_count.get(word, 0) + 1

            # Handle duplicate words by creating unique versions
            for word, count in words_count.items():
                if count > 1:
                    cur_word = word
                    for i in range(count - 1):
                        cur_word = f"{cur_word} {word}"
                    words_index[cur_word] = words_index[word]
                    words_index.pop(word)

            # Sort words by frequency
            words = [x[0] for x in sorted(words_index.items(), key=lambda x: x[1])]
            words_len = len(words)
            words = ' '.join(words).split()

            # Insert words into tree
            for index, value in enumerate(words):
                no_cutting = 0
                if rebuild == 1:
                    no_cutting = 1
                elif index <= float(len(words)) * CUTTING_PERCENT:
                    no_cutting = 1

                is_end = 1 if index == words_len - 1 else 0
                self.tree_list[pid].insert_node(
                    words[:index], value, para, is_end_node=is_end, 
                    no_cutting=no_cutting, rebuild=rebuild
                )

    def do(self, logs: List[Tuple[str, List[str]]], para: Dict) -> Dict[str, List[Tuple[str]]]:
        """Main method to process logs and extract templates.
        
        Args:
            logs: List of (pid, words) tuples
            para: Parameters dictionary
            
        Returns:
            Dictionary of {pid: list_of_templates}
        """
        if not logs:
            return {}

        # Initialize data structures
        pids = set()
        words_frequency = {}

        # Process logs and count word frequencies
        for pid, words in logs:
            pids.add(pid)
            for w in words:
                words_frequency[w] = words_frequency.get(w, 0) + 1

        # Sort words by frequency (descending)
        words_frequency = sorted(words_frequency.items(), 
                               key=lambda x: (x[1], x[0]), 
                               reverse=True)
        words_frequency = [x[0] for x in words_frequency]

        # Save word frequencies
        with open(para['fre_word_path'], 'w') as f:
            for w in words_frequency:
                f.write(f"{w}\n")

        # Build the frequency tree
        self._init_tree(list(pids))
        self.auto_temp(logs, words_frequency, para)

        # Visualize tree if requested
        if para['plot_flag'] == 1:
            self.drawTree()

        # Extract templates from tree
        all_paths = {}
        for pid in self.tree_list:
            all_paths[pid] = []
            path = self.traversal_tree(self.tree_list[pid])
            for template in path[1]:
                all_paths[pid].append(template)
            # Sort templates by length (longest first)
            all_paths[pid].sort(key=lambda x: len(x), reverse=True)

        # Save templates
        with open(para['template_path'], 'w') as f:
            template_counter = 1
            for pid, templates in all_paths.items():
                for path in templates:
                    f.write(f"{pid} {' '.join(path)}\n")
                    template_counter += 1

        return all_paths

    def drawTree(self) -> None:
        """Visualize the tree using pygraphviz."""
        try:
            import pygraphviz as pgv
            A = pgv.AGraph(directed=True, strict=True)
            unique_dir = {}  # Record word occurrences for unique IDs

            for pid in self.tree_list:
                head_node = self.tree_list[pid]._head
                myQueue = [head_node]

                while myQueue:
                    node = myQueue.pop(0)
                    cur_data = node.get_data()
                    cur_father = f"{cur_data}{' '*node._index}"

                    for child_node in node.get_children():
                        myQueue.append(child_node)
                        cur_child = child_node.get_data()
                        
                        # Handle duplicate words
                        unique_dir[cur_child] = unique_dir.get(cur_child, 0) + 1
                        child_node._index = unique_dir[cur_child] - 1
                        cur_child = f"{cur_child}{' '*child_node._index}"

                        if cur_father:
                            # Color end nodes blue and pruned nodes red
                            if child_node.is_end_node:
                                A.add_node(cur_child, color='blue')
                            elif child_node._change_to_leaf:
                                A.add_node(cur_child, color='red')
                            else:
                                A.add_node(cur_child)

                            A.add_node(cur_father)
                            A.add_edge(cur_father, cur_child)

            A.write('tree.dot')
            A.layout('dot')
            A.draw('tree.png')
        except ImportError:
            print("Warning: pygraphviz not available - tree visualization disabled")


def getMsgFromNewSyslog(log: str, msg_id_index: int = 3) -> Tuple[str, List[str]]:
    """Parse a log line into (pid, words) tuple.
    
    Args:
        log: Raw log line
        msg_id_index: Not currently used
        
    Returns:
        Tuple of (empty_pid, list_of_words)
    """
    msg_list = log.split()
    if len(msg_list) > 300:  # Limit very long logs
        msg_list = msg_list[:300]
    return ('', msg_list)


def getLogsAndSave(para: Dict) -> None:
    """Main function to process log file and generate templates.
    
    Args:
        para: Dictionary of parameters
    """
    short_log = 0
    log_once_list = []
    wft = WordsFrequencyTree()

    with open(para['data_path']) as IN:
        for log in IN:
            log = log.strip()
            if not log:
                continue
            
            pid, words = getMsgFromNewSyslog(log)
            if len(words) < para['short_threshold']:
                short_log += 1
                continue
            log_once_list.append((pid, words))

    print('Creating templates...')
    wft.do(log_once_list, para)
    print(f'Filtered {short_log} short logs (threshold = {para["short_threshold"]})')
    print(f'Templates saved to: {para["template_path"]}')
    print(f'Word frequencies saved to: {para["fre_word_path"]}')

def parse_logs(input_path: str, output_template: str, output_freq: str, 
               leaf_num: int = 4, short_threshold: int = 5) -> None:
    """Main function to parse logs using FT-Tree algorithm."""
    para = {
        'FIRST_COL': 0,
        'NO_CUTTING': 1,
        'CUTTING_PERCENT': 0.3,
        'data_path': input_path,
        'template_path': output_template,
        'fre_word_path': output_freq,
        'leaf_num': leaf_num,
        'short_threshold': short_threshold,
        'plot_flag': 0
    }

    # First count total lines for progress bar
    with open(input_path) as f:
        total_lines = sum(1 for _ in f)

    # Initialize processing with progress bar
    log_once_list = []
    short_log = 0
    wft = WordsFrequencyTree()

    with click.progressbar(length=total_lines, label='Processing logs') as bar:
        with open(input_path) as IN:
            for log in IN:
                log = log.strip()
                if not log:
                    bar.update(1)
                    continue
                
                pid, words = getMsgFromNewSyslog(log)
                if len(words) < short_threshold:
                    short_log += 1
                    bar.update(1)
                    continue
                
                log_once_list.append((pid, words))
                bar.update(1)

    print('\nCreating templates...')
    wft.do(log_once_list, para)
    print(f'Filtered {short_log} short logs (threshold = {short_threshold})')
    print(f'Templates saved to: {output_template}')
    print(f'Word frequencies saved to: {output_freq}')

    