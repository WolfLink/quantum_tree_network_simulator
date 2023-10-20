import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def name_for_node(l, i):
    return f"{l}_{i}"

ROUTER_COLOR = "grey"

def color_for_layer(l, n):
    l += 4 - n
    if l == 0:
        return "coral"
    if l == 1:
        return "gold"
    if l == 2:
        return "skyblue"
    if l == 3:
        return "mediumseagreen"

def color_for_layer_blue(l):
    if l == 0:
        return "royalblue"
    if l == 1:
        return "cornflowerblue"
    if l == 2:
        return "lightblue"
    if l == 3:
        return "lightsteelblue"

def a(k):
    if k > 2:
        return 1 / np.sin(np.pi / k)
    else:
        return 2

def position_top(k, l, i, n, gridish=False):
    position = [0,0]
    #print(f"computing position for {k}**{n}: {l}, {i}")
    ak = a(k)
    if l > 0:
        branch_angle = 0
        for l2 in range(1,l+1):
            branch_direction = (i // (k**(l-l2))) % k
            if gridish:
                if branch_angle == 0:
                    branch_angle = np.pi * 0.32
                else:
                    branch_angle = np.pi * 0.18
                branch_angle = branch_angle + np.pi + branch_direction * 2 * np.pi / k # + np.pi / k # 0 gives a space filling but redundant tree.  pi/k gives a very symmetric but overlapping tree.  pi*0.23 was chosen by trial and error for a 4,4 tree.
            else:
                branch_angle = branch_angle + np.pi + branch_direction * 2 * np.pi / k + np.pi * 0.23 # + np.pi / k # 0 gives a space filling but redundant tree.  pi/k gives a very symmetric but overlapping tree.  pi*0.23 was chosen by trial and error for a 4,4 tree.
            #branch_angle = np.mod(branch_angle, 2*np.pi)
            branch_length = ak**(n-l2)
            #print(f"Branch length for {k} {l2} is {branch_length}")
            
            position[0] += np.sin(branch_angle) * branch_length
            position[1] += np.cos(branch_angle) * branch_length
    return np.array(position)


def position_side(k, l, i, n):
    position = [0,0]
    #print(f"computing position for {k}**{n}: {l}, {i}")
    ak = a(k)
    if l > 0:
        for l2 in range(1,l+1):
            branch_direction = (i // (k**(l-l2))) % k
            branch_length = ak**(n-l2)

            branch_spread = k**(n-l2)

            position[0] += branch_direction * branch_spread / (k-1) - branch_spread / 2
            position[1] -= branch_length
            
    return np.array(position)

def gen_tree(k, n, layout_style="top", draw_routers=True, minedge=1, minnode=5, edge_scaling_dims=2):
    graph = nx.Graph()
    output_data = {
            "graph" : graph,
            "position" : dict(),
            "layer" : dict(),
            "labels" : dict(),
            "nodesize" : [],
            "nodelist" : [],
            "nodecolor" : [],
            "edgelist" : [],
            "edgewidth" : [],
            }

    for l in range(n):
        for i in range(k**l):
            node_name = name_for_node(l, i)
            node_layer = l

            output_data["labels"][node_name] = node_name
            if layout_style == "top":
                output_data["position"][node_name] = position_top(k, l, i, n)
            elif layout_style == "top_gridish":
                output_data["position"][node_name] = position_top(k, l, i, n, gridish=True)
            elif layout_style == "side":
                output_data["position"][node_name] = position_side(k, l, i, n)

            output_data["layer"][node_name] = node_layer

            output_data["nodesize"].append(k**(n - l) * minnode)
            output_data["nodelist"].append(node_name)
            output_data["nodecolor"].append(color_for_layer(l, n))

            # add the edge
            if l > 0:
                graph.add_edge(name_for_node(l-1, i // k), node_name)
                output_data["edgelist"].append((name_for_node(l-1, i // k), node_name))
                output_data["edgewidth"].append((np.sqrt(k) if edge_scaling_dims == 3 else k)**(n-l - 1)  * minedge)

                if draw_routers:
                    routersize = output_data["nodesize"][-1] * 0.95
                    num_routers = 2 ** (n-l-1) - 1
                    start = output_data["position"][node_name]
                    end = output_data["position"][name_for_node(l-1, i // k)]

                    current_pos = [start[0], start[1]]
                    dx = (end[0] - start[0]) / (num_routers + 1)
                    dy = (end[1] - start[1]) / (num_routers + 1)
                    for j in range(num_routers):
                        current_pos = [current_pos[0] + dx, current_pos[1] + dy]
                        router_name = f"router_{l}_{i}_{j}"
                        graph.add_node(router_name)

                        output_data["position"][router_name] = current_pos
                        output_data["labels"][router_name] = "R"

                        output_data["nodelist"].append(router_name)
                        output_data["nodesize"].append(routersize)
                        output_data["nodecolor"].append(ROUTER_COLOR)



                #graph.add_node(node_name)
    return output_data


def gen_grid(width, height, edgewidth=4, nodesize=300):
    graph = nx.Graph()

    output_data = {
            "graph" : graph,
            "position" : dict(),
            "layer" : dict(),
            "labels" : dict(),
            "nodesize" : [],
            "nodelist" : [],
            "nodecolor" : [],
            "edgelist" : [],
            "edgewidth" : [],
            }

    for x in range(width):
        for y in range(height):
            node_id = (x,y)
            node_name = f"{x}, {y}"
            
            output_data["labels"][node_id] = node_name
            output_data["layer"][node_id] = 0
            output_data["nodesize"].append(nodesize)
            output_data["nodelist"].append(node_id)
            output_data["nodecolor"].append("grey")
            output_data["position"][node_id] = [x, y]

            if x > 0:
                graph.add_edge((x - 1, y), (x,y))
                output_data["edgelist"].append(((x - 1, y), (x,y)))
                output_data["edgewidth"].append(edgewidth)
            if y > 0:
                graph.add_edge((x, y - 1), (x,y))
                output_data["edgelist"].append(((x, y - 1), (x,y)))
                output_data["edgewidth"].append(edgewidth)

    return output_data


def make_plot(data, size=5, outfile=None):
    fig = plt.figure(figsize=(size,size))
    #nx.draw_networkx_nodes(data["graph"], data["position"], node_size=data["size"])
    nodes = nx.draw_networkx_nodes(data["graph"], data["position"], nodelist=data["nodelist"], node_size=data["nodesize"], node_color=data["nodecolor"], edgecolors="black", linewidths=1)
    #nx.draw_networkx_labels(data["graph"], data["position"], data["labels"])
    edges = nx.draw_networkx_edges(data["graph"], data["position"], edgelist=data["edgelist"], width=data["edgewidth"])
    return (nodes, edges, fig, data)

# make_plot(gen_tree(4,4, layout_style="side", draw_routers=False, minedge=0.5), outfile="side_nr.pdf")
# make_plot(gen_tree(4,3, layout_style="top_gridish", draw_routers=False, minedge=2, minnode=11), outfile="top.pdf", size=2.5)
# make_plot(gen_tree(4,4, layout_style="side", minedge=0.75, minnode=3.5, edge_scaling_dims=3), outfile="side.pdf")
# make_plot(gen_grid(4, 4), outfile="grid.pdf", size=2.5)
# 
# make_plot(gen_tree(5,3, layout_style="top", draw_routers=False, minedge=2, minnode=10), outfile="pentagon.pdf")
# make_plot(gen_grid(16,16), outfile="biggrid.pdf", size=2.5)


