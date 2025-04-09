import networkx as nx
from assign_stereotype_label import compute_ancestry, get_dominating_keys

def add_ancestry_labels(input_dot_file, output_dot_file):
    """
    Read the input DOT file, compute ancestry for each node, and create a new DOT file
    with node declarations and xlabels based on the ancestry.
    """
    try:
        # Read the input DOT file
        G = nx.drawing.nx_pydot.read_dot(input_dot_file)
        
        # Get dominating keys and compute ancestry
        dominating_keys = get_dominating_keys(G)
        ancestry_dict = compute_ancestry(G, dominating_keys)
        
        # Create new DOT content with node declarations and xlabels
        new_dot_content = ['digraph inferenceModel{']
        
        # Get nodes in topological order
        nodes_topo = list(nx.topological_sort(G))
        
        # Process nodes in topological order
        for node in nodes_topo:
            if node in ancestry_dict:
                # Get all incoming edges and sort them
                in_edges = list(G.in_edges(node, data=True))
                
                # Sort edges: first by perc/actu label, then by source node's topological order
                def sort_key(edge):
                    source, _, data = edge
                    label = data.get('label', '')
                    # perc edges come before actu edges
                    label_priority = 0 if 'perc' in label else 1
                    return (label_priority, nodes_topo.index(source))
                
                sorted_edges = sorted(in_edges, key=sort_key)
                
                # Add all incoming edges to this node
                for source, _, data in sorted_edges:
                    label = data.get('label', '')
                    if label:
                        # Remove any existing quotes from the label
                        label = label.strip('"\'')
                        new_dot_content.append(f'    {source} -> {node}[label="{label}"]')
                    else:
                        new_dot_content.append(f'    {source} -> {node}')
                
                # Convert ancestry set to string, ensuring node itself is first
                anc_list = list(ancestry_dict[node])
                if node in anc_list:
                    anc_list.remove(node)
                    anc_list.insert(0, node)
                ancestry_str = str(ancestry_dict[node])
                
                # Add node declaration
                new_dot_content.append(f'    {node} [xlabel="{ancestry_str}"];')
                
                # Add a blank line after each node's section for readability
                new_dot_content.append('')
        
        new_dot_content.append('}')
        
        # Write the new DOT file
        with open(output_dot_file, 'w') as f:
            f.write('\n'.join(new_dot_content))
            
        print(f"Successfully created {output_dot_file} with ancestry labels")
        
    except Exception as e:
        print(f"Error processing DOT files: {e}")
        raise

if __name__ == "__main__":
    input_file = "process_dot/stereotype_graphvis_input.dot"
    output_file = "process_dot/stereotype_graphvis_output.dot"
    add_ancestry_labels(input_file, output_file) 