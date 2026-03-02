# Dreath-first search (DFS) is an algorithm for traversing or searching 
# tree or graph data structures. The algorithm starts at the root node 
# (selecting some arbitrary node as the root in the case of a graph) 
# and explores as far as possible along each branch before backtracking.


graph = {
  '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : ['8'],
  '8' : []
}

visited = set()

def dfs(visited, graph, node):
    if node not in visited: #Check if the node has not been visited
        print(node) #Print the node which is being visited
        visited.add(node) #Add node to the set of visited nodes
        for neighbour in graph: # Loop through the neighbours of the node
            dfs(visited, graph, neighbour) # recursively visit each neighbor
            
print("Following is the Depth-First Search")
dfs(visited, graph, '2')            


          
# numbers = {
#   5: 3,
#   3: 2,
#   7: 8,
#   2: 0,
# }

# # Iterating through the dictionary and adding 5 to each value
# for i in numbers:
#   # numbers[i] += 5
#   print(numbers[i])
          
# graph['5'] = ['3', '7']
# for neighbor in graph['5']:  # neighbor = '3', then '7'
#     print(neighbor)