# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 00:42:45 2022

@author: -
"""
import numpy as np 



def get_block_diagonal_matrix(repeat, matrix):
    return np.kron(np.eye(repeat,dtype=int),matrix)
    

def find_index(string, array):
    return np.where(array==string)[0][0]
    
def add_edge_adjacency(index1, index2, adjacency):
    adjacency[index1][index2]=1
    adjacency[index2][index1]=1
    return adjacency


def get_minimal_nodes_position(x=0.0,y=0.0, x_offset=1.0, y_offset=1.0):
    nodes_position = np.zeros((3,2))
    nodes_position[0][0] = x
    nodes_position[0][1] = y
    nodes_position[1][0] = x-x_offset
    nodes_position[1][1] = y-y_offset
    nodes_position[2][0] = x+x_offset
    nodes_position[2][1] = y-y_offset
    return nodes_position
    
def offset_positions(nodes_positions, x_offset, y_offset):
    nodes_positions = nodes_positions - [x_offset, y_offset]
    return nodes_positions

def make_connections_adjacency(new_adjacency, new_state_num, new_state_string):
    zero=''
    one=''
    two=''
    for i in range(len(new_state_num[0])-1):
        zero+='0'
        one+='1'
        two+='2'
    str1=zero+'1'
    str2=zero+'2'
    index1 = find_index(str1, new_state_string)
    index2 = find_index(str2, new_state_string)
    new_adjacency= add_edge_adjacency(index1, index2, new_adjacency)
    str1=one+'0'
    str2=one+'2'
    index1 = find_index(str1, new_state_string)
    index2 = find_index(str2, new_state_string)
    new_adjacency= add_edge_adjacency(index1, index2, new_adjacency)
    str1=two+'0'
    str2=two+'1'
    index1 = find_index(str1, new_state_string)
    index2 = find_index(str2, new_state_string)
    new_adjacency= add_edge_adjacency(index1, index2, new_adjacency)
    return new_adjacency

def hanoi_graph_increment(state_num, state_string, adjacency):
    temp_1 = state_num.copy()
    temp_block1 = np.append(temp_1, 0*np.ones((len(state_string),1), dtype=int), axis=1)
    temp_2 = (state_num.copy()+1)%3
    temp_block2 = np.append(temp_2, 2*np.ones((len(state_string),1), dtype=int), axis=1)
    temp_3 = (state_num.copy()+2)%3
    temp_block3 = np.append(temp_3, 1*np.ones((len(state_string),1), dtype=int), axis=1)
    
    new_state_num=temp_block1
    new_state_num=np.append(new_state_num, temp_block3, axis=0)
    new_state_num=np.append(new_state_num, temp_block2, axis=0)
    new_adjacency = get_block_diagonal_matrix(3, adjacency)
    temp_string=new_state_num.astype(str)
    new_state_string = [''.join(st)  for st in temp_string]
    new_state_string = np.array(new_state_string).reshape((len(new_state_string),1))
    
    #print(new_state_num)
    #print(new_state_string)
    #print(new_adjacency)
    new_adjacency=make_connections_adjacency(new_adjacency, new_state_num, new_state_string)
    return new_state_num, new_state_string, new_adjacency


def compute_state_and_adjacency(number_of_cheeses):
    state_num = np.zeros((3,1), dtype=int)
    state_string=np.zeros((3,1), dtype=str)
    for i in range(3):
        state_num[i]=i
        state_string[i]=str(i)
    adjacency=np.ones((3,3), dtype= int)
    adjacency[0][0]=0
    adjacency[1][1]=0
    adjacency[2][2]=0
    nodes_pos=get_minimal_nodes_position()
    
    if number_of_cheeses!=1:
        base_x_offset=1.0
        base_y_offset=1.0
        for i in range(number_of_cheeses-1):
            state_num, state_string, adjacency = hanoi_graph_increment(state_num, state_string, adjacency)
            nodes_pos_1 = nodes_pos.copy()
            nodes_pos_2 = nodes_pos.copy()
            nodes_pos_3 = nodes_pos.copy()
            offset_factor = 2**(i+1)
            x_offset=offset_factor*base_x_offset
            y_offset=offset_factor*base_y_offset
            nodes_pos_2 = offset_positions(nodes_pos_2, x_offset, y_offset)
            x_offset=-1.0*x_offset
            nodes_pos_3 = offset_positions(nodes_pos_3, x_offset, y_offset)
            nodes_pos = nodes_pos_1
            if i%2==1:
                nodes_pos=np.append(nodes_pos, nodes_pos_2, axis=0)
                nodes_pos=np.append(nodes_pos, nodes_pos_3, axis=0)
            else:
                nodes_pos=np.append(nodes_pos, nodes_pos_3, axis=0)
                nodes_pos=np.append(nodes_pos, nodes_pos_2, axis=0)   
    return state_num, state_string, adjacency, nodes_pos



if __name__ == "__main__":
    state_num, state_string, adjacency, nodes_pos = compute_state_and_adjacency(5)
    data_dict = {'state_string': state_string, 'adjacency': adjacency, 'nodes_pos': nodes_pos}
    #np.save('states_and_adjacency_5_3.npy', data_dict)
    import scipy.io as sio
    # Save the data in a .mat file
    sio.savemat('states_and_adjacency_5_3.mat', data_dict)
    print(state_num)
    print(state_string)
    print(adjacency)
