
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
sns.set_theme()


def nodes_generator(df):
    G = nx.Graph()  
    for i in range(1,len(df)+1):
        j = [i]
        x = df[1][i]
        y = df[2][i]
        coord = (x,y)
        G.add_nodes_from(j,  pos = coord)
        del j
    pos=nx.get_node_attributes(G,'pos')
    return G,pos

def edges_generator(G,df):
    

    for i in range(1,len(df)):
        a = df[0][i-1]
        b = df[0][i]
        G.add_edge(a, b)

    return G


def graph_plotter(G,pos,name,size):

    color_map = ['red' if node == 1 else '#00b4d9' for node in G] 
    fig, ax = plt.subplots()
    fig.set_size_inches(size, 0.75*size)
    nx.draw_networkx(G, pos,with_labels=True, node_size=120, font_weight='bold',node_color=color_map, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(name +' configuration')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
