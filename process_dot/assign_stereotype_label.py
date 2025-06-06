import networkx as nx
import sys

def compute_ancestry(G, dominating_keys):
    """
    Compute the conditional ancestry for each node in a DAG according to the following rules:
    
      0. Every node's ancestry is represented as a set.
      1. By default, each node's ancestry contains itself.
      2. For a node with one or more parents, its ancestry is computed by inheriting:
           - Each parent's previously computed ancestry, and
           - The direct parent node's identifier itself (if its name does not contain "classification").
      3. If a parent's name contains the substring "classification" (case insensitive),
         then that parent's own identifier is excluded (removed from its ancestry) when inherited.
      4. Dominating keys: If any direct parent is in the `dominating_keys` set, then the node's ancestry
         becomes exactly that parent's ancestry (note: the node itself is not added in this case).
    """
    ancestry = {}
    for node in nx.topological_sort(G):
        parents = list(G.predecessors(node))
        
        if not parents:
            if node in dominating_keys:
                ancestry[node] = set()
            else:
                ancestry[node] = {node}
        else:
            dominating_parent = None
            for p in parents:
                if p in dominating_keys:
                    dominating_parent = p
                    break
                    
            if dominating_parent is not None:
                ancestry[node] = ancestry[dominating_parent].copy()
            else:
                contributions = set()
                for p in parents:
                    parent_ancestry = ancestry[p].copy()
                    if "classification" in p.lower():
                        parent_ancestry.discard(p)
                    contributions = contributions.union(parent_ancestry)
                    
                    if "classification" not in p.lower():
                        contributions.add(p)
                # Always add the node itself to its ancestry when inheriting from non-dominating parents
                ancestry[node] = contributions.union({node})
    return ancestry

def get_dominating_keys(G):
    """Get the set of dominating keys based on the specified rules."""
    dominating_keys = set()
    
    # Add nodes containing "attributes" except "harmful_attributes" and classification nodes
    for node in G.nodes():
        if ("attributes" in node and 
            "harmful_attributes" not in node and 
            "classification" not in node.lower()):
            dominating_keys.add(node)
            
    # Add nodes containing "attribution_form" except "attribution_form" and classification nodes
    for node in G.nodes():
        if ("attribution_form" in node and 
            node != "attribution_form" and 
            "classification" not in node.lower()):
            dominating_keys.add(node)
            
    # Add nodes containing "target_groups" except "sensitive_target_groups" and classification nodes
    for node in G.nodes():
        if ("target_groups" in node and 
            "sensitive_target_groups" not in node and 
            "classification" not in node.lower()):
            dominating_keys.add(node)
            
    # Add nodes containing "adopting_subjects" except "abnormal_adopting_subjects" and classification nodes
    for node in G.nodes():
        if ("adopting_subjects" in node and 
            "abnormal_adopting_subjects" not in node and 
            "classification" not in node.lower()):
            dominating_keys.add(node)
            
    return dominating_keys

def main():
    # Read the DOT file
    dot_file = "process_dot/stereotype_graphvis_input.dot"
    try:
        # First read the DOT file as a string to check its contents
        with open(dot_file, 'r') as f:
            dot_content = f.read()
            print("DOT file contents:")
            print(dot_content[:500] + "...")  # Print first 500 chars
        
        # Now read it with NetworkX
        G = nx.drawing.nx_pydot.read_dot(dot_file)
        
        # Print graph information
        print("\nGraph information:")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        print("\nNodes in graph:")
        for node in sorted(G.nodes()):
            print(f"- {node}")
        
        # Get dominating keys
        dominating_keys = get_dominating_keys(G)
        
        # Print dominating keys
        print("\nDominating Keys:")
        for key in sorted(dominating_keys):
            print(f"- {key}")
        
        # Compute ancestry
        ancestry_dict = compute_ancestry(G, dominating_keys)
        
        # Print ancestry for each node
        print("\nAncestry for each node:")
        for node, anc in sorted(ancestry_dict.items()):
            # Convert set to list and ensure node itself is first
            anc_list = list(anc)
            if node in anc_list:
                anc_list.remove(node)
                anc_list.insert(0, node)
            print(f"{node}: {anc_list}")
            
    except Exception as e:
        print(f"Error processing DOT file '{dot_file}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 