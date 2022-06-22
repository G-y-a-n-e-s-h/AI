#!/usr/bin/env python
# coding: utf-8

# In[1]:


# BFS algorithm

import collections
def bfs(graph, root):

    visited, queue = set(), collections.deque([root])
    visited.add(root)

    while queue:

        # Dequeue a vertex from queue
        vertex = queue.popleft()
        print(str(vertex) + " ", end="")

        # If not visited, mark it as visited, and
        # enqueue it

        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)


if __name__ == '__main__':
    graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
    print("Following is Breadth First Traversal: ")
    bfs(graph, 0)


# In[2]:


# DFS algorithm

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    print(start)

    for next in graph[start] - visited:
        dfs(graph, next, visited)
    return visited


graph = {'0': set(['1', '2']),
         '1': set(['0', '3', '4']),
         '2': set(['0']),
         '3': set(['1']),
         '4': set(['2', '3'])}

dfs(graph, '0')


# In[3]:


#Best First Seach Algorithm

from queue import PriorityQueue
vertices = 14
graph = [[] for i in range(vertices)]

# Function For Implementing Best First Search
# Gives output path having lowest cost

def best_first_search(Source, target, vertices):
	visited = [0] * vertices
	pq = PriorityQueue()
	pq.put((0,Source))
	print("path:")
	while not pq.empty():
		u = pq.get()[1]

		# Displaying the path having lowest cost

		print(u, end=" ")
		if u == target:
			break

		for v, c in graph[u]:
			if not visited[v]:
				visited[v] = True
				pq.put((c, v))
	print()

# Function for adding edges to graph

def addedge(x, y, cost):
	graph[x].append((y, cost))
	graph[y].append((x, cost))

# The nodes shown in above example(by alphabets) are
# implemented using integers addedge(x,y,cost);

addedge(0, 1, 1)
addedge(0, 2, 8)
addedge(1, 2, 12)
addedge(1, 4, 13)
addedge(2, 3, 6)
addedge(4, 3, 3)
source = 0
target = 2
best_first_search(source, target, vertices)

# This code is contributed by Jyotheeswar Ganne


# In[4]:


#Beam First Search

from math import log
from numpy import array
from numpy import argmax
# beam search
def beam_search_decoder(data, k):
        sequences = [[list(), 0.0]]
        # walk over each step in sequence
        for row in data:
                        all_candidates = list()
                        # expand each current candidate
                        for i in range(len(sequences)):
                                    seq, score = sequences[i]
                                    for j in range(len(row)):
                                            candidate = [seq + [j], score - log(row[j])]
                                            all_candidates.append(candidate)
                        # order all candidates by score
                        ordered = sorted(all_candidates, key=lambda tup:tup[1])
                        # select k best
                        sequences = ordered[:k]
        return sequences
# define a sequence of 10 words over a vocab of 5 words
data = [[0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.5, 0.4, 0.3, 0.2, 0.1]]
data = array(data)
# decode sequence
result = beam_search_decoder(data, 3)
# print result
for seq in result:
            print(seq)


# In[61]:


#A* Algorithm

from collections import deque
class Graph:
            # example of adjacency list (or rather map)
            # adjacency_list = {
            # 'A': [('B', 1), ('C', 3), ('D', 7)],
            # 'B': [('D', 5)],
            # 'C': [('D', 12)]
            # }
            def __init__(self, adjacency_list):
                   self.adjacency_list = adjacency_list
            def get_neighbors(self, v):
                    return self.adjacency_list[v]
            # heuristic function with equal values for all nodes
            def h(self, n):
                H = {
                            'A': 1,
                            'B': 1,
                            'C': 1,
                            'D': 1
                        }
                return H[n]
            def a_star_algorithm(self, start_node, stop_node):
                        # open_list is a list of nodes which have been visited, but who's neighbors
                        # haven't all been inspected, starts off with the start node
                        # closed_list is a list of nodes which have been visited
                        # and who's neighbors have been inspected
                        open_list = set([start_node])
                        closed_list = set([])
                        # g contains current distances from start_node to all other nodes
                        # the default value (if it's not found in the map) is +infinity
                        g = {}
                        g[start_node] = 0
                        # parents contains an adjacency map of all nodes
                        parents = {}
                        parents[start_node] = start_node
                        while len(open_list) > 0:
                                n = None
                                # find a node with the lowest value of f() - evaluation function
                                for v in open_list:
                                    if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                                            n = v;
                                if n == None:
                                        print('Path does not exist!')
                                        return None
                                # if the current node is the stop_node
                                # then we begin reconstructin the path from it to the start_node
                                if n == stop_node:
                                        reconst_path = []
                                        while parents[n] != n:
                                                    reconst_path.append(n)
                                                    n = parents[n]
                                        reconst_path.append(start_node)
                                        reconst_path.reverse()
                                        print('Path found: {}'.format(reconst_path))
                                        return reconst_path
                                # for all neighbors of the current node do
                                for (m, weight) in self.get_neighbors(n):
                                # if the current node isn't in both open_list and closed_list
                                # add it to open_list and note n as it's parent
                                    if m not in open_list and m not in closed_list:
                                             open_list.add(m)
                                             parents[m] = n
                                             g[m] = g[n] + weight
                                            # otherwise, check if it's quicker to first visit n, then m
                                            # and if it is, update parent data and g data
                                            # and if the node was in the closed_list, move it to open_list
                                    else:
                                            if g[m] > g[n] + weight:
                                                g[m] = g[n] + weight
                                                parents[m] = n
                                            if m in closed_list:
                                                closed_list.remove(m)
                                                open_list.add(m)
                                                # remove n from the open_list, and add it to closed_list
                                                # because all of his neighbors were inspected
                                open_list.remove(n)
                                closed_list.add(n)
                        print('Path does not exist!')
                        return None
        
adjacency_list = {
'A': [('B', 1), ('C', 3), ('D', 7)],
'B': [('D', 5)],
'C': [('D', 12)]
}
graph1 = Graph(adjacency_list)
graph1.a_star_algorithm('A', 'D')


# In[64]:


#NLP 

# Importing Libraries
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
# Downloading Some Necessary Packages
nltk.download('stopwords')
nltk.download('punkt')
paragraph = """I have three visions for India. In 3000 years of our history, people from all over the
world have come and invaded us, captured our lands, conquered our minds. From Alexander
onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all
of them came and looted us, took over what was ours. Yet we have not done this to any other
nation. We have not conquered anyone. We have not grabbed their land, their culture, their history
and tried to enforce our way of life on them. Why? Because we respect the freedom of others. That
is why my first vision is that of freedom. I believe that India got its first vision of this in 1857,
when we started the War of Independence. It is this freedom that we must protect and nurture and
build on. If we are not free, no one will respect us. My second vision for India’s development. For
fifty years we have been a developing nation. It is time we see ourselves as a developed nation. We
are among the top 5 nations of the world in terms of GDP. We have a 10 percent growth rate in
most areas. Our poverty levels are falling. Our achievements are being globally recognised today.
Yet we lack the self-confidence to see ourselves as a developed nation, self-reliant and selfassured.
Isn’t this incorrect? I have a third vision. India must stand up to the world. Because I
believe that unless India stands up to the world, no one will respect us. Only strength respects
strength. We must be strong not only as a military power but also as an economic power. Both
must go hand-in-hand. My good fortune was to have worked with three great minds. Dr. Vikram
Sarabhai of the Dept. of space, Professor Satish Dhawan, who succeeded him and Dr. Brahm
Prakash, father of nuclear material. I was lucky to have worked with all three of them closely and
consider this the great opportunity of my life. I see four milestones in my career"""
# Tokenizing Using PorterStemmer
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()
for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
        sentences[i] = ' '.join(words)
        # Tokenizing Using SnowballStemmer
        stemm = SnowballStemmer('english')
for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        words = [stemm.stem(word) for word in words if word not in set(stopwords.words('english'))]
        sentences[i] = ' '.join(words)
        # Removing Punctuation
        filtered = []
        s = set(string.punctuation) # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
for i in range(len(sentences)):
        words = nltk.word_tokenize(sentences[i])
        for j in words:
            if j not in s :
                filtered.append(j);
print(filtered)


# In[71]:


#Branch and Bound

import threading as th
from copy import deepcopy
class Node:
            def __init__(self, parent=None, state=[]):
                    self.parent = parent
                    self.generator_lock = th.Lock()
                    self.generator = self._child_gen()
                    self.state = state
            def _child_gen(self):
                    for i in range(1, 4):
                            state = deepcopy(self.state) + [i]
                            yield Node(self, state)
            def next_child(self):
                with self.generator_lock:
                    return next(self.generator, None)
            def is_leaf(self):
                    return len(self.state) >= 10
            def __repr__(self):
                    return '<Node state="{}">'.format(self.state)
class Worker:
            def __init__(self, id, searcher):
                    self.searcher = searcher # type: Searcher
                    self.id = id
            def __call__(self):
                    print("start worker: {}".format(self.id))
                    while not self.searcher.is_end():
                        self._run()
                    print("end worker: {}".format(self.id))
            def _run(self):
                    node = self.searcher.get_last_node()
                    if node is None:
                        return
                    if node.is_leaf():
                        self.searcher.remove_node(node)
                        self.searcher.add_result(node)
                        return
                    bounds = self.searcher.get_bounds()
                    if not self.satisfy_bounds(node, bounds):
                        self.searcher.remove_node(node)
                        return
                    child = node.next_child()
                    if child is None:
                        self.searcher.remove_node(node)
                    else:
                        self.searcher.add_node(child)
            def satisfy_bounds(self, node, bound):
                    return True
class Searcher:
            def __init__(self):
                self.root_node = Node()
                self.nodes = [self.root_node] # TODO: priority queue
                self.nodes_lock = th.Lock()
                self._is_end = False
                self.workers = [ Worker(i, self) for i in range(8) ]
                self.results = set()
                self.results_lock = th.Lock()
                self.bounds = [None, None]
                self.bounds_lock = th.Lock()
                self.threads = []
            def run(self):
                self.threads = [
                        th.Thread(target=w, name="thread:{}".format(idx))
                        for idx, w in enumerate(self.workers)
                        ]
                for t in self.threads:
                            t.start()
                for t in self.threads:
                            t.join()
            def get_last_node(self):
                with self.nodes_lock:
                    if self.nodes:
                        return self.nodes[-1]
                    else:
                        self._is_end = True
                        return None
            def add_node(self, node):
                with self.nodes_lock:
                    self.nodes.append(node)
            def remove_node(self, node):
                with self.nodes_lock:
                    if node in self.nodes:
                        self.nodes.remove(node)
            def is_end(self):
                return self._is_end
            def check_end(self):
                with self.nodes_lock:
                    self._is_end = len(self.nodes) == 0
            def add_result(self, node):
                with self.results_lock:
                    self.results.add(node)
            def get_bounds(self):
                with self.bounds_lock:
                    return deepcopy(self.bounds)
def main():
        s = Searcher()
        s.run()
        print(len(s.results))
        assert len(s.results) == 3**10
if __name__ == '__main__':
    main()


# In[82]:


#Tic Tac Toe

import random
class TicTacToe:
    def __init__(self):
        self.board = []
    def create_board(self):
        for i in range(3):
            row = []
            for j in range(3):
                row.append('-')
            self.board.append(row)
    def get_random_first_player(self):
        return random.randint(0, 1)
    def fix_spot(self, row, col, player):
        self.board[row][col] = player
    def is_player_win(self, player):
                win = None
                n = len(self.board)
                # checking rows
                for i in range(n):
                    win = True
                    for j in range(n):
                        if self.board[i][j] != player:
                            win = False
                            break
                    if win:
                        return win
                # checking columns
                for i in range(n):
                    win = True
                    for j in range(n):
                        if self.board[j][i] != player:
                            win = False
                            break
                if win:
                    return win
                # checking diagonals
                win = True
                for i in range(n):
                    if self.board[i][i] != player:
                        win = False
                        break
                if win:
                    return win
                win = True
                for i in range(n):
                        if self.board[i][n - 1 - i] != player:
                            win = False
                            break
                if win:
                    return win
                return False
                for row in self.board:
                    for item in row:
                        if item == '-':
                            return False
                    return True
    def is_board_filled(self):
                    for row in self.board:
                        for item in row:
                            if item == '-':
                                return False
                        return True
    def swap_player_turn(self, player):
                        return 'X' if player == 'O' else 'O'
    def show_board(self):
                for row in self.board:
                    for item in row:
                        print(item, end=" ")
                    print()
    def start(self):
                        self.create_board()
                        player = 'X' if self.get_random_first_player() == 1 else 'O'
                        while True:
                            print(f"Player {player} turn")
                            self.show_board()
                            # taking user input
                            row, col = list(
                                    map(int, input("Enter row and column numbers to fix spot: ").split()))
                            print()
                            # fixing the spot
                            self.fix_spot(row - 1, col - 1, player)
                            # checking whether current player is won or not
                            if self.is_player_win(player):
                                print(f"Player {player} wins the game!")
                                break
                            # checking whether the game is draw or not
                            if self.is_board_filled():
                                print("Match Draw!")
                                break
                            # swapping the turn
                            player = self.swap_player_turn(player)
                            # showing the final view of board
                            print()
                            self.show_board()
                            # starting the game 
tic_tac_toe = TicTacToe()
tic_tac_toe.start()


# In[ ]:





# In[ ]:




