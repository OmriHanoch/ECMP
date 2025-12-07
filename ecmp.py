import networkx as nx
import math
import matplotlib.pyplot as plt
import argparse
import random
import sys
import numpy as np

# --- 1. Fat-Tree Topology Functions (From your original code) ---

def calculate_fat_tree_params(k):
    """Calculates the structural parameters for a k-port Fat-tree."""
    if k % 2 != 0 or k < 2:
        raise ValueError("k must be an even number greater than or equal to 2.")

    k_half = k // 2
    
    params = {
        'k': k,
        'k_half': k_half,
        'num_pods': k,
        'num_core_switches': k_half * k_half,
        'total_hosts': (k**3) // 4
    }
    return params

def build_fat_tree(k, link_capacity=1.0):
    """Builds the Fat-tree topology as a NetworkX graph with initial load attributes."""
    try:
        params = calculate_fat_tree_params(k)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    G = nx.Graph() 
    k_half = params['k_half']
    core_switches = []
    pod_switches_data = {} 

    # 1. Create Nodes
    for c_idx in range(params['num_core_switches']):
        node_id = f"C_{c_idx}"
        G.add_node(node_id, type='core', pod=-1) 
        core_switches.append(node_id)

    current_host_idx = 0
    for pod_id in range(params['num_pods']):
        pod_switches_data[pod_id] = {'agg': [], 'edge': [], 'host_to_edge': {}}
        
        # Aggregation Switches
        for a_idx in range(k_half):
            node_id = f"A_{pod_id}_{a_idx}"
            G.add_node(node_id, type='agg', pod=pod_id)
            pod_switches_data[pod_id]['agg'].append(node_id)

        # Edge Switches and Hosts
        for e_idx in range(k_half):
            edge_node_id = f"E_{pod_id}_{e_idx}"
            G.add_node(edge_node_id, type='edge', pod=pod_id)
            pod_switches_data[pod_id]['edge'].append(edge_node_id)

            for h_idx in range(k_half):
                host_node_id = f"H_{current_host_idx}"
                G.add_node(host_node_id, type='host', pod=pod_id, edge_anchor=edge_node_id)
                # Add edge with initial load and capacity
                G.add_edge(edge_node_id, host_node_id, layer='edge_host', capacity=link_capacity, load_uv=0.0, load_vu=0.0)
                
                pod_switches_data[pod_id]['host_to_edge'][host_node_id] = edge_node_id
                current_host_idx += 1

    # 2. Create Edges
    for pod_id in range(params['num_pods']):
        agg_switches = pod_switches_data[pod_id]['agg']
        edge_switches = pod_switches_data[pod_id]['edge']

        # A. Aggregation <-> Edge
        for agg_node in agg_switches:
            for edge_node in edge_switches:
                G.add_edge(agg_node, edge_node, layer='agg_edge', capacity=link_capacity, load_uv=0.0, load_vu=0.0)

        # B. Core <-> Aggregation (Striping)
        for agg_idx, agg_node in enumerate(agg_switches):
            strip_start_index = agg_idx * k_half
            for core_offset in range(k_half):
                core_idx = strip_start_index + core_offset
                core_node = core_switches[core_idx]
                G.add_edge(agg_node, core_node, layer='agg_core', capacity=link_capacity, load_uv=0.0, load_vu=0.0)
                
    G.graph['params'] = params
    G.graph['pod_data'] = pod_switches_data
    return G

# --- 2. ECMP Routing and Allocation ---

def allocate_flow_with_ecmp(G, source, target, flow_demand):
    """
    Finds all shortest paths (ECMP paths) and allocates flow demand 
    to one path chosen by a simple random hash (Static Hashing).
    """
    
    try:
        # 1. Find all shortest paths (ECMP paths)
        # This function returns a generator of lists of nodes (the path)
        all_ecmp_paths = list(nx.all_shortest_paths(G, source=source, target=target))
    except nx.NetworkXNoPath:
        # No path exists, skip the flow
        return False, 0
    
    if not all_ecmp_paths:
        return False, 0 # Should not happen in a connected Fat-Tree, but for safety

    # 2. ECMP Hashing: Choose one path randomly (simulating a perfect hash function)
    # The choice of path represents the outcome of the static hash function.
    chosen_path = random.choice(all_ecmp_paths)
    
    # 3. Allocate demand to the chosen path
    # Iterate over the path (node-by-node) and update the edge load for the correct direction
    
    for i in range(len(chosen_path) - 1):
        u = chosen_path[i]
        v = chosen_path[i+1]
        
        # NetworkX stores edges sorted (u, v) or (v, u). 
        # We need to ensure we update the load attribute corresponding to the direction u -> v.
        
        # Check if (u, v) exists, otherwise it's (v, u)
        if G.has_edge(u, v):
            # Check the order of nodes in the graph's edge representation
            # NetworkX can store an edge as (A, B) even if the user added (B, A) later.
            # We will use the canonical edge key (u, v) where u < v lexicographically 
            # for reliable access, but this is complicated.
            
            # SIMPLER APPROACH: Direct Attribute Access (assuming load_uv is u->v)
            # Since the graph is undirected, we check which node is 'smaller' lexicographically
            
            if u < v:
                G.edges[u, v]['load_uv'] += flow_demand
            else: # v < u
                G.edges[v, u]['load_vu'] += flow_demand # Note: load_vu is load for v -> u
        # If the edge was removed due to failure (not relevant here, but good practice):
        # else: continue 
        
    return True, len(chosen_path) - 1

# --- 3. Traffic Scenario Generators ---

def generate_traffic_data(G, scenario_type, num_flows, link_capacity):
    """Generates flow demands for the two required scenarios."""
    host_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'host']
    
    flows = [] # List of tuples: (source, target, demand)
    
    # הגדרת מכפיל העומס הכולל (כדי להבטיח אתגר לרשת)
    LOAD_FACTOR = 10 

    if scenario_type == 'SUCCESS_A':
        # Scenario A: Many small flows to achieve good statistical balancing
        
        base_demand = link_capacity / num_flows * LOAD_FACTOR 
        
        for _ in range(num_flows):
            src = random.choice(host_nodes)
            dst = random.choice(host_nodes)
            # Ensure src and dst are different and not in the same pod for inter-pod traffic focus
            while src == dst or G.nodes[src]['pod'] == G.nodes[dst]['pod']:
                 dst = random.choice(host_nodes)
            
            # Demand is almost uniform for maximum balancing effect
            demand = base_demand * (1 + random.uniform(-0.1, 0.1)) 
            flows.append((src, dst, demand))
            
    elif scenario_type == 'FAILURE_B':
        # Scenario B: Elephant Flows cause congestion due to RANDOM collisions (natural failure)
        
        # --- פרמטרים מעודכנים לכישלון טבעי ---
        # הגדלת שיעור הפילים מ-5% ל-15% כדי להבטיח קולז' סטטיסטי
        elephant_flow_rate = 0.15 
        num_elephant_flows = int(num_flows * elephant_flow_rate) 
        num_mouse_flows = num_flows - num_elephant_flows
        
        # Total demand is the same as Scenario A, but concentrated
        total_demand = link_capacity * LOAD_FACTOR
        
        # 1. Elephant Flows (90% of total load)
        elephant_load = total_demand * 0.90
        elephant_demand_per_flow = elephant_load / num_elephant_flows
        
        # *** שינוי קריטי: בוחרים מקור ויעד רנדומלי לחלוטין (כמו ב-A) ***
        for _ in range(num_elephant_flows):
            src = random.choice(host_nodes)
            dst = random.choice(host_nodes)
            while src == dst or G.nodes[src]['pod'] == G.nodes[dst]['pod']:
                 dst = random.choice(host_nodes)
                 
            flows.append((src, dst, elephant_demand_per_flow))
            
        # 2. Mouse Flows (10% of total load)
        mouse_load = total_demand * 0.10
        mouse_demand_per_flow = mouse_load / num_mouse_flows

        for _ in range(num_mouse_flows):
            src = random.choice(host_nodes)
            dst = random.choice(host_nodes)
            while src == dst:
                 dst = random.choice(host_nodes)
                 
            flows.append((src, dst, mouse_demand_per_flow))

    # Shuffle the flows to simulate real-world arrival
    random.shuffle(flows)
    return flows

def run_simulation(G, flows):
    """Iterates through flows and allocates them via ECMP."""
    total_flow_demand = sum(demand for _, _, demand in flows)
    print(f"Total Flow Demand: {total_flow_demand:.2f}")
    
    successful_flows = 0
    
    for src, dst, demand in flows:
        success, _ = allocate_flow_with_ecmp(G, src, dst, demand)
        if success:
            successful_flows += 1
            
    print(f"Successfully allocated flows: {successful_flows}/{len(flows)}")

    
def reset_graph_load(G):
    """Resets the load on all edges in the graph."""
    for u, v, data in G.edges(data=True):
        data['load_uv'] = 0.0
        data['load_vu'] = 0.0

# --- 4. Visualization ---

def visualize_load(G, k, scenario_name):
    """Draws the Fat-tree and colors edges based on utilization."""
    
    # 1. Calculate Utilization and Edge Colors
    edge_colors = []
    edges_to_draw = []
    utilizations = []
    
    # Iterate over all edges and consider both directions for visualization
    for u, v, data in G.edges(data=True):
        capacity = data['capacity']
        
        # Direction 1: u -> v
        load_uv = data['load_uv']
        utilization_uv = min(load_uv / capacity, 1.5) # Cap utilization at 150%
        
        if load_uv > 0:
            edges_to_draw.append((u, v))
            # Use QUADRATIC scaling to increase sensitivity to small differences
            # This makes the color more responsive to small load variations
            color_value = (utilization_uv / 1.5) ** 0.5  # Square root scaling for better differentiation
            edge_colors.append(plt.cm.RdYlGn_r(color_value))
            utilizations.append(utilization_uv)
            
        # Direction 2: v -> u
        load_vu = data['load_vu']
        utilization_vu = min(load_vu / capacity, 1.5) # Cap utilization at 150%

        if load_vu > 0 and load_uv == 0: # Avoid drawing the same edge twice
             max_util = max(utilization_uv, utilization_vu)
             
             if (u,v) not in edges_to_draw: 
                 edges_to_draw.append((u, v))
                 color_value = (max_util / 1.5) ** 0.5  # Square root scaling
                 edge_colors.append(plt.cm.RdYlGn_r(color_value))
                 utilizations.append(max_util)
    
    # If no flows, just draw the basic graph
    if not edge_colors:
        edges_to_draw = G.edges()
        edge_colors = 'gray'

    # 2. Define Custom Hierarchical Layout
    pos = {}
    y_core, y_agg, y_edge, y_host = 3.0, 2.0, 1.0, 0.0
    k_half = k // 2
    
    core_nodes = sorted([n for n, attr in G.nodes(data=True) if attr['type'] == 'core'])
    core_x_spacing = 2.0 / (len(core_nodes) + 1)
    for i, node_id in enumerate(core_nodes):
        pos[node_id] = (i * core_x_spacing - 1 + core_x_spacing/2, y_core)
    
    pod_width_scale = 1.0 / k 
    
    for pod_id in range(k):
        pod_center_x = (pod_id + 0.5) * pod_width_scale * 2 - 1 
        
        agg_nodes = sorted([n for n, attr in G.nodes(data=True) if attr['type'] == 'agg' and attr['pod'] == pod_id])
        agg_x_spacing = pod_width_scale / (len(agg_nodes) + 1) * 2 
        for i, node_id in enumerate(agg_nodes):
            pos[node_id] = (pod_center_x - pod_width_scale + (i + 0.5) * agg_x_spacing, y_agg)

        edge_nodes = sorted([n for n, attr in G.nodes(data=True) if attr['type'] == 'edge' and attr['pod'] == pod_id])
        edge_x_spacing = pod_width_scale / (len(edge_nodes) + 1) * 2
        for i, node_id in enumerate(edge_nodes):
            pos[node_id] = (pod_center_x - pod_width_scale + (i + 0.5) * edge_x_spacing, y_edge)
            
        hosts_per_edge = k_half
        all_hosts_in_pod_data = [ (n, attr) for n, attr in G.nodes(data=True) 
                                 if attr['type'] == 'host' and attr['pod'] == pod_id]
        
        all_hosts_in_pod_data.sort(key=lambda item: int(item[0].split('_')[1])) 
        
        for e_idx, edge_node_id in enumerate(edge_nodes):
            start_index = e_idx * hosts_per_edge
            hosts_in_this_edge_range_data = all_hosts_in_pod_data[start_index : start_index + hosts_per_edge]
            
            edge_x_pos = pos[edge_node_id][0]
            host_sub_x_spacing = 0.1 / k_half 
            
            for i, (host_node_id, attr) in enumerate(hosts_in_this_edge_range_data):
                  pos[host_node_id] = (edge_x_pos - (hosts_per_edge - 1) * host_sub_x_spacing / 2 + i * host_sub_x_spacing, y_host)

    # 3. Draw
    plt.figure(figsize=(14, 10)) 
    color_map = {'host': '#CCCCCC', 'edge': '#FFC04D', 'agg': '#92D050', 'core': '#9CC2E5'}
    node_colors = [color_map[attr['type']] for n, attr in G.nodes(data=True)]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color=edge_colors, width=2.0)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')

    # Add color bar (legend) - better formatted
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=0.0, vmax=1.5))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', pad=0.02, label='Link Utilization (%)')
    
    # Set custom tick labels: 0%, 50%, 100%, 150%
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5])
    cbar.set_ticklabels(['0%\n(Empty)', '50%\n(Light)', '100%\n(Full)', '150%\n(Congestion)'])
    
    plt.title(f"Fat-Tree (K={k}) - ECMP Performance: {scenario_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(description="Simulate ECMP performance on a k-ary Fat-Tree network.")
    
    parser.add_argument('--K-VALUE', type=int, default=4, help="The constant K value for the Fat-Tree.")
    parser.add_argument('--FLOWS', type=int, default=1000, help="Total number of flows to simulate.")
    parser.add_argument('--CAPACITY', type=float, default=1.0, help="Link capacity (used as the 100 baseline).")
    parser.add_argument('--SEED', type=int, default=43, help="Random seed for reproducibility.")

    args = parser.parse_args()
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    
    K = args.K_VALUE
    
    # 1. Build the Fat-Tree topology
    print(f"\n--- Building Fat-Tree Topology (K={K}) ---")
    G = build_fat_tree(K, link_capacity=args.CAPACITY)
    params = calculate_fat_tree_params(K)
    print(f"Total Hosts: {params['total_hosts']}, Total Edges: {G.number_of_edges()}")
    # --- 2. Run Scenario A: SUCCESS ---
    print("\n--- Running Scenario A: SUCCESS (Near-Perfect Load Balancing) ---")
    flows_A = generate_traffic_data(G, 'SUCCESS_A', args.FLOWS, args.CAPACITY)
    run_simulation(G, flows_A)
    visualize_load(G, K, "Scenario A: Near-Perfect Load Balancing (Many Small Flows)")
    #print(flows_A)

    # Reset load for next scenario
    reset_graph_load(G)
    
    # --- 3. Run Scenario B: FAILURE ---
    print("\n--- Running Scenario B: FAILURE (Congestion due to Collisions) ---")
    flows_B = generate_traffic_data(G, 'FAILURE_B', args.FLOWS, args.CAPACITY)
    run_simulation(G, flows_B)
    visualize_load(G, K, "Scenario B: ECMP Failure (Few Elephant Flows Collision)")
    #print(flows_B)

if __name__ == "__main__":
    main()
