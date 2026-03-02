# Graph represented as adjacency list (dictionary)
# Each key is a node, and its value is a list of connected neighbors
graph = {
  '5' : ['3','7'],   # Node '5' connects to nodes '3' and '7'
  '3' : ['2', '4'],  # Node '3' connects to nodes '2' and '4'
  '7' : ['8'],       # Node '7' connects to node '8'
  '2' : [],          # Node '2' has no outgoing connections
  '4' : ['8'],       # Node '4' connects to node '8'
  '8' : []           # Node '8' has no outgoing connections
}

visited = []  # List to keep track of nodes we've already visited (prevents revisiting)
queue = []    # Queue to store nodes waiting to be explored (FIFO - First In First Out)

def bfs(visited, graph, node):  # BFS function takes visited list, graph, and starting node
  visited.append(node)          # Mark the starting node as visited
  queue.append(node)            # Add starting node to the queue

  while queue:                  # Keep looping while there are nodes in the queue
    m = queue.pop(0)            # Remove and get the first node from queue (FIFO behavior)
    print(m, end=" ")           # Print the current node being visited

    for neighbour in graph[m]:          # Loop through all neighbors of current node
      if neighbour not in visited:      # Only process if neighbor hasn't been visited
        visited.append(neighbour)       # Mark neighbor as visited
        queue.append(neighbour)         # Add neighbor to queue for later exploration

# Driver Code - This is where the program execution starts
print("Following is the Breadth-First Search")
bfs(visited, graph, '5')    # Call BFS starting from node '5'