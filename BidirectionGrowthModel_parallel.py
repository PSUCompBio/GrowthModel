import random
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
import matplotlib as mpl
import scipy.io as sio
import time
import threading
import Queue

#  start of the program
# Define the initial positions
def initial_pos(cell_no, rad, Z):
    # 0    1    2    3        4        5       6       7        8              9          10
    # r  theta  z  dir_r  dir_theta  dir_z  cell_no  step  previous_node  current_node  branch_no
    pos0 = elevenCol
    for i in range(cell_no):
        pos_initial = np.array([[rad * random.uniform(0, 1), 2 * math.pi * random.uniform(0, 1),
                                 rad * random.uniform(-1, 0) + Z, 0, 0, 1, i, 0, 0, 0, 1]])
        pos0 = np.vstack((pos0, pos_initial))
        reset_nodes = elevenCol
        if np.array_equal(reset_nodes, pos0[0]):
            pos0 = np.delete(pos0, 0, 0)
    return pos0



def initial_nodes(cell_no):
    count_nodes = np.array([0])
    for i in range(cell_no):
        initial_node = np.array([0])
        count_nodes = np.vstack((count_nodes, initial_node))
        if i == 0:
            reset_nodes = np.array([0])
            if np.array_equal(reset_nodes, count_nodes[0]):
                count_nodes = np.delete(count_nodes, 0, 0)
    return count_nodes



def initial_nodes_b(cell_no):
    count_nodes = np.array([0])
    for i in range(cell_no):
        initial_node_b = np.array([0])
        count_nodes = np.vstack((count_nodes, initial_node_b))
        if i == 0:
            reset_nodes = np.array([0])
            if np.array_equal(reset_nodes, count_nodes[0]):
                count_nodes = np.delete(count_nodes, 0, 0)
    return count_nodes



# Calculate the concentration gradients
def gradient(pos_previous, pos_current):
    #Calculate the concentration C
    cos_sum = 1 + 2 * (np.cos(pos_previous[:, 1] - pos_current[1])) * np.exp(-k3)
    for i in range(2, 41):
        cos_sum += 2 * np.cos(i * (pos_previous[:, 1] - pos_current[1])) * np.exp(-k3 * i ** 2)
    sin_sum = -2 * (np.sin(pos_previous[:, 1] - pos_current[1])) * np.exp(-k3)
    for i in range(2, 41):
        sin_sum += -2 * i * np.sin(i * (pos_previous[:, 1] - pos_current[1])) * np.exp(-k3 * i ** 2)
    C = sp.j0(pos_previous[:, 0] - pos_current[0]) * cos_sum * np.exp(
        -((pos_previous[:, 2] - pos_current[2]) ** 2) / k5)

    # Calculate the gradients
    grad = threeCol
    # gradient in r direction
    posx = pos_previous[:, 0] - pos_current[0]
    Grad_2 = -1 * sp.jv(1, posx)
    grad[0] = np.sum(Grad_2)
    # gradient in theta direction
    grad[1] = np.sum((sp.j0(pos_previous[:, 0] - pos_current[0]) / rad) * sin_sum * np.exp(
        -((pos_previous[:, 2] - pos_current[2]) ** 2) / k5))
    # gradient in z direction
    grad[2] = np.sum(C * (-2) * (pos_previous[:, 1] - pos_current[1]) / k5)
    return grad



#Find the new position in forward direction
def generate_next(step_no, pos_current, pos_previous, v_bi,q):
    pos_new_full = elevenCol
    i = 0
    current_length = len(pos_current)

    while i < current_length:

        step = v0 * random.uniform(0.8, 1.2)
        dir1 = np.array(pos_current[i][3:6])
        grad = gradient(pos_previous[:, 0:3], pos_current[i][0:3])
        err1 = np.array([random.uniform(-1.0, 1.0), random.gauss(err1_theta, err1_theta), random.gauss(0.0, 1)])

        strength = math.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
        if strength > 0.001:
            grad1 = grad / strength
        else:
            grad1 = threeCol
            grad = threeCol

        cos = np.sum(grad1 * dir1)
        if cos > 0.0 or np.array_equal(grad1, np.array([0., 0., 0.])):
            grad[1] = grad[1] / rad
            #dir2 = dir1 + s1 * (grad1 + err1)
            dir2 = dir1 + s1 * grad1 + err1
        else:
            # re-orient the pull direction in the direction of growth
            R = dir1[1] * grad1[2] - dir1[2] * grad1[1]
            R = 1.0 - 2 * int(R < 0.0)
            grad1[1:3] = (-cos * dir1[1:3] + math.sqrt(1 - cos ** 2) * np.array([-R * dir1[2], R * dir1[1]]))
            #dir2 = dir1 + s1 * (grad1 + err1)
            dir2 = dir1 + s1 * grad1 + err1

        l1 = math.sqrt(dir2[0] ** 2 + dir2[1] ** 2 + dir2[2] ** 2)
        dir2 = dir2 / l1

        l = (v_bi * (v0_grad * strength + step)) * 2 ** (-step_no / tau)

        dir2[1] = dir2[1] / rad
        pos2 = pos_current[i][0:3] + dir2 * l

        # NEW ADDING:Make sure the r direction is in the tube (in the range of rad)
        cell_number = pos_current[i][6]
        node_counter = initial_node[cell_number]

        # Check overlap of tips(only check tips' r,theta,z), ingore the middle section

        # make uTENN grow out after reaching the end
        if pos2[2] <= z_bw:
            if abs(pos2[0]) <= rad:
                pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
                initial_node[cell_number] = node_counter + 1
                pos_new_full = np.vstack((pos_new_full, pos2_full))
                reset_nodes = elevenCol
                if np.array_equal(reset_nodes, pos_new_full[0]):
                    pos_new_full = np.delete(pos_new_full, 0, 0)
            else:
                sign = np.sign(pos2[0])
                pos2[0] = sign * rad
                pos2[1] = pos_current[i][1]
                pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
                initial_node[cell_number] = node_counter + 1
                pos_new_full = np.vstack((pos_new_full, pos2_full))
                reset_nodes = elevenCol
                if np.array_equal(reset_nodes, pos_new_full[0]):
                    pos_new_full = np.delete(pos_new_full, 0, 0)
        else:
            pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
            initial_node[cell_number] = node_counter + 1
            pos_new_full = np.vstack((pos_new_full, pos2_full))
            reset_nodes = elevenCol
            if np.array_equal(reset_nodes, pos_new_full[0]):
                pos_new_full = np.delete(pos_new_full, 0, 0)

        #uTENN stay inside the tube forever
        # if abs(pos2[0]) <= rad:
        #     pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
        #     initial_node[cell_number] = node_counter + 1
        #     pos_new_full = np.vstack((pos_new_full, pos2_full))
        #     reset_nodes = elevenCol
        #     if np.array_equal(reset_nodes, pos_new_full[0]):
        #         pos_new_full = np.delete(pos_new_full, 0, 0)
        # else:
        #     sign = np.sign(pos2[0])
        #     pos2[0] = sign * rad
        #     pos2[1] = pos_current[i][1]
        #     pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter + 1, pos_current[i][10]))
        #     initial_node[cell_number] = node_counter + 1
        #     pos_new_full = np.vstack((pos_new_full, pos2_full))
        #     reset_nodes = elevenCol
        #     if np.array_equal(reset_nodes, pos_new_full[0]):
        #         pos_new_full = np.delete(pos_new_full, 0, 0)
        i += 1
    #return pos_new_full
    q.put(pos_new_full)



#Find the new position in backward direction
def generate_next_back(step_no, pos_current, pos_previous, v_bi,q):
    pos_new_full = elevenCol
    i = 0
    current_length = len(pos_current)
    while i < current_length:
        step = v0 * random.uniform(0.8, 1.2)
        dir1 = np.array(pos_current[i][3:6])
        grad = gradient(pos_previous[:, 0:3], pos_current[i][0:3])
        err1 = np.array([random.uniform(-1.0, 1.0), random.gauss(err1_theta, err1_theta), random.gauss(0.0, 1)])

        strength = math.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
        if strength > 0.001:
            grad1 = grad / strength
        else:
            grad1 = threeCol
            grad = threeCol
        cos = np.sum(grad1 * dir1)

        if cos > 0.0 or np.array_equal(grad1, np.array([0., 0., 0.])):
            grad[1] = grad[1] / rad
            #dir2 = dir1 + s2 * (grad1 + err1)
            dir2 = dir1 + s2 * grad1 + err1
        else:
            # re-orient the pull direction in the direction of growth
            R = dir1[1] * grad1[2] - dir1[2] * grad1[1]
            R = 1.0 - 2 * int(R < 0.0)
            grad1[1:3] = (-cos * dir1[1:3] + math.sqrt(1 - cos ** 2) * np.array([-R * dir1[2], R * dir1[1]]))
            #dir2 = dir1 + s2 * (grad1 + err1)
            dir2 = dir1 + s2 * grad1 + err1

        l1 = math.sqrt(dir2[0] ** 2 + dir2[1] ** 2 + dir2[2] ** 2)
        dir2 = dir2 / l1
        l = (v_bi * (v0_grad * strength + step)) * 2 ** (-step_no / tau)  # Remember to change back

        dir2[1] = dir2[1] / rad
        pos2 = pos_current[i][0:3] - dir2 * l

        cell_number = pos_current[i][6]
        node_counter_b = initial_node_b[cell_number]

        # make uTENN grow out after reaching the end
        if pos2[2] >= z_fwd:
            if abs(pos2[0]) <= rad:
                pos2_full = np.hstack(
                    (pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter_b + 1, pos_current[i][10]))
                initial_node_b[cell_number] = node_counter_b + 1

                pos_new_full = np.vstack((pos_new_full, pos2_full))
                reset_nodes = elevenCol
                if np.array_equal(reset_nodes, pos_new_full[0]):
                    pos_new_full = np.delete(pos_new_full, 0, 0)
            else:
                sign = np.sign(pos2[0])
                pos2[0] = sign * rad
                pos2[1] = pos_current[i][1]

                pos2_full = np.hstack(
                    (pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter_b + 1, pos_current[i][10]))
                initial_node_b[cell_number] = node_counter_b + 1

                pos_new_full = np.vstack((pos_new_full, pos2_full))
                reset_nodes = elevenCol
                if np.array_equal(reset_nodes, pos_new_full[0]):
                    pos_new_full = np.delete(pos_new_full, 0, 0)

        else:
            pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter_b + 1, pos_current[i][10]))
            initial_node_b[cell_number] = node_counter_b + 1
            pos_new_full = np.vstack((pos_new_full, pos2_full))
            reset_nodes = elevenCol
            if np.array_equal(reset_nodes, pos_new_full[0]):
                pos_new_full = np.delete(pos_new_full, 0, 0)

        # uTENN stay inside the tube forever
        # if abs(pos2[0]) <= rad:
        #     pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter_b + 1, pos_current[i][10]))
        #     initial_node_b[cell_number] = node_counter_b + 1
        #
        #     pos_new_full = np.vstack((pos_new_full, pos2_full))
        #     reset_nodes = elevenCol
        #     if np.array_equal(reset_nodes, pos_new_full[0]):
        #         pos_new_full = np.delete(pos_new_full, 0, 0)
        # else:
        #     sign = np.sign(pos2[0])
        #     pos2[0] = sign * rad
        #     pos2[1] = pos_current[i][1]
        #
        #     pos2_full = np.hstack((pos2, dir2, pos_current[i][6], step_no, pos_current[i][9], node_counter_b + 1, pos_current[i][10]))
        #     initial_node_b[cell_number] = node_counter_b + 1
        #
        #     pos_new_full = np.vstack((pos_new_full, pos2_full))
        #     reset_nodes = elevenCol
        #     if np.array_equal(reset_nodes, pos_new_full[0]):
        #         pos_new_full = np.delete(pos_new_full, 0, 0)

        i += 1
    #return pos_new_full
    q.put(pos_new_full)



def plotter(pos_all, pos_all_b):
    fig1 = plt.figure(figsize=(20, 15))
    ax3d = fig1.add_subplot(111, projection='3d')

    theta = pos_all[:, 1]
    x = pos_all[:, 0] * np.cos(theta)
    y = pos_all[:, 0] * np.sin(theta)
    z = pos_all[:, 2]
    ax3d.plot(x, y, z, 'r+')

    theta_b = pos_all_b[:, 1]
    x_b = pos_all_b[:, 0] * np.cos(theta_b)
    y_b = pos_all_b[:, 0] * np.sin(theta_b)
    z_b = pos_all_b[:, 2]
    ax3d.plot(x_b, y_b, z_b, 'g+')

    # theta_init = pos0[:, 1]
    # x_init = pos0[:, 0] * np.cos(theta_init)
    # y_init = pos0[:, 0] * np.sin(theta_init)
    # z_init = pos0[:, 2]
    # ax3d.plot(x_init, y_init, z_init, 'ro')

    # theta_init_b = pos0_b[:, 1]
    # x_init_b = pos0_b[:, 0] * np.cos(theta_init_b)
    # y_init_b = pos0_b[:, 0] * np.sin(theta_init_b)
    # z_init_b = pos0_b[:, 2]
    # ax3d.plot(x_init_b, y_init_b, z_init_b, 'go')

    #ax3d.set_aspect(3.75)
    ax3d.set_xlim([-200, 200])
    ax3d.set_ylim([-200, 200])
    ax3d.set_zlim([-200, z_bw])
    ax3d.set_xlabel(r'$ \mu m$', fontsize=15, labelpad=20)
    ax3d.set_ylabel(r'$ \mu m$', fontsize=15, labelpad=20)
    ax3d.set_zlabel(r'$ \mu m$', fontsize=15, labelpad=30)
    ax3d.set_zticks(np.arange(0, z_bw, 100))
    ax3d.set_xticks(np.arange(-200, 200, 100))
    ax3d.set_yticks(np.arange(-200, 200, 100))
    ax3d.set_title('%.2f Days' % (step_no * 0.03), fontsize=30)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='vertical')
    ax3d.view_init(elev=4, azim=315)
    #plt.savefig('Step_%d.png' % (step_no + 1), dpi=400, bbox_inches='tight', pad_inches=1)
    plt.show()


def pos_saver(pos_all, pos_all_b):
    #np.savetxt('pos_fwd_step_%d.txt' % (step_no + 1), pos_all)
    #np.savetxt('pos_bwd_step_%d.txt' % (step_no + 1), pos_all_b)

    name_file_fwd = 'pos_bi_fwd_' + str(z_bw) + '_' + str(cell_no)
    name_file_bw = 'pos_bi_bw_' + str(z_bw) + '_' + str(cell_no_b)
    name_file_fwd_txt = name_file_fwd + '.txt'
    name_file_bw_txt = name_file_bw + '.txt'
    name_file_fwd_mat = name_file_fwd + '.mat'
    name_file_bw_mat = name_file_bw + '.mat'

    np.savetxt(name_file_fwd_txt, pos_all)
    np.savetxt(name_file_bw_txt, pos_all_b)

    #sio.savemat(name_file_fwd_mat, {'pos_all': pos_all})
    #sio.savemat(name_file_bw_mat, {'pos_all_b': pos_all_b})



# Find the Growth Rate
def GrowthRate(pos_current, pos_current_b):
    z_current = pos_current[:, 2]
    z_current_b = pos_current_b[:, 2]
    # z_max = np.hstack((step_no, np.amax(z_current), np.amin(z_current_b)))  # backward min is the longest tip
    # z_growth = np.vstack((z_growth, z_max))
    z_mean = np.hstack((step_no, np.average(z_current), np.average(z_current_b)))
    return z_mean


def GrowthRateSaver(z_growth):
    growth_rate = np.array([0, 0])
    z = z_growth[:, 1]
    for n in range(0, len(z) - 1):
        growth = (z[n + 1] - z[n]) / 0.03
        time = 0.03 * n
        growth_full = np.hstack((time, growth))
        growth_rate = np.vstack((growth_rate, growth_full))
        reset_nodes = np.array([0, 0])
        if np.array_equal(reset_nodes, growth_rate[0]):
            growth_rate = np.delete(growth_rate, 0, 0)
    np.savetxt('Bi_growthrate_50cells.txt', growth_rate)
    plt.plot(growth_rate[:, 0], growth_rate[:, 1])
    plt.ylabel('growth rate')
    plt.title('Bidirection')
    # plt.savefig('growthrate2.png', dpi=400, bbox_inches='tight', pad_inches=1)
    plt.show()





# Define all the parameters
k0 = 5
k3 = 0.001
k5 = 1
s1 = 5.0e-2             # Sensitivity to concentration gradients from forward direction
s2 = 5.0e-2             # Sensitivity to concentration gradients from backward direction

v0 = 15                 # Base growth rate
v0_grad = 0.008         # Growth rate based on gradient strength
tau = 150.0             # Time constant for growth rate
err1_theta =  0.785     # Direction perturbation term in theta
v_bi = 1                # Chemical effect of bidirectional growth: 1 at beginning

tau_b = 20              # Time constant for branching rate
P = 0                   # Branching probability at infinite: 0 means no branching
B1 = 0.1                # Branching condition 1: Branching may happen if probability value greater than B1
B2 = 0.15               # Branching condition 2: Branching may happen if random uniform term greater than B2

total_step = 10         # Total simulation step: each step is 0.03 days
cell_no = 500            # Total cell number in forward direction
cell_no_b = cell_no     # Total cell number in backward direction

rad = 50.0              # Inner radius of microcolumn
z_fwd = 0               # Initial seeding position in z direction for forward growth
z_bw = 500              # Initial seeding position in z direction for backward growth (length of the tube)

#Create zeros
elevenCol = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
threeCol = np.array([0, 0, 0])
z_growth = threeCol
branch_step = 0
branch_step_b = 0



starttime = time.clock()


step_no = 1
pos0 = initial_pos(cell_no, rad, z_fwd)
pos0_b = initial_pos(cell_no, rad, z_bw)

initial_node = initial_nodes(cell_no)
initial_node_b = initial_nodes_b(cell_no_b)

# From step 0 to step 1, pos_current = pos_previous
# pos_1 = generate_next(step_no, pos0, pos0, v_bi)
q1_fwd = Queue.Queue()
t1_fwd = threading.Thread(target=generate_next, args=(step_no, pos0, pos0, v_bi, q1_fwd))
t1_fwd.daemon = True
t1_fwd.start()
pos_1 = q1_fwd.get()

# After step 1, pos_current different from pos_previous
pos_all = np.vstack((pos0, pos_1))
pos_current = pos_1
pos_previous = pos0
#pos_previous = pos_1  # NEED DISCUSSION

# From step 0 to step 1, backward growth
#pos_1b = generate_next_back(step_no, pos0_b, pos0_b, v_bi)
q1_bw = Queue.Queue()
t1_bw = threading.Thread(target=generate_next, args=(step_no, pos0_b, pos0_b, v_bi, q1_bw))
t1_bw.daemon = True
t1_bw.start()
pos_1b = q1_bw.get()

pos_all_b = np.vstack((pos0_b, pos_1b))
pos_current_b = pos_1b
pos_previous_b = pos0_b
#pos_previous_b = pos_1b  # NEED DISCUSSION



for step_no in range(2, total_step):
    print step_no
    original_length = len(pos_current)
    a = 0
    while a < len(pos_current):
        p = P * (1 - math.exp(-(step_no - branch_step) * 0.3 / (tau_b)))
        if (p >= B1 and random.uniform(0, 1) > B2):
            insert_array = np.hstack((pos_current[a][0:10], 2))
            pos_current = np.insert(pos_current, a + 1, insert_array, 0)
            a += 1
        a += 1
    if len(pos_current) > original_length:
        branch_step = step_no
    else:
        branch_step = branch_step



    counter_fwd = 0
    total_core = 2
    qlist_fwd = [Queue.Queue() for i in range(total_core)]
    unit_fwd = len(pos_current) // total_core
    vlist_fwd = []
    for n_core in range(total_core):
        start = n_core * unit_fwd
        end = start + unit_fwd

        q_fwd = qlist_fwd[n_core]
        t_new_fwd = threading.Thread(target=generate_next, args=(step_no, pos_current[start:end], pos_previous, v_bi, q_fwd))
        t_new_fwd.daemon = True
        t_new_fwd.start()
        # print q.get()
        vlist_fwd.append(q_fwd.get())

    pos_new = np.vstack(vlist_fwd)

    #pos_new = generate_next(step_no, pos_current, pos_previous, v_bi)
    pos_all = np.vstack((pos_all, pos_new))

    for m in range(len(pos_new)):
        pos_new[m] = np.hstack((pos_new[m][0:10], 1))

    pos_previous = pos_current
    # pos_previous = pos_new  #NEED DISCUSSION
    pos_current = pos_new

    ######################################################################
    original_length_b = len(pos_current_b)
    a_b = 0
    while a_b < len(pos_current_b):
        p_b = P * (1 - math.exp(-(step_no - branch_step_b) * 0.3 / (tau_b)))
        if (p_b >= B1 and random.uniform(0, 1) > B2):
            insert_array_b = np.hstack((pos_current_b[a_b][0:10], 2))
            pos_current_b = np.insert(pos_current_b, a_b + 1, insert_array_b, 0)
            a_b += 1
        a_b += 1
    if len(pos_current_b) > original_length_b:
        branch_step_b = step_no
    else:
        branch_step_b = branch_step_b

    counter_bw = 0
    total_core = 2
    qlist_bw = [Queue.Queue() for i in range(total_core)]
    unit_bw = len(pos_current_b) // total_core
    vlist_bw = []
    for n_core in range(total_core):
        start = n_core * unit_bw
        end = start + unit_bw

        q_bw = qlist_bw[n_core]
        t_new_bw = threading.Thread(target=generate_next, args=(step_no, pos_current_b[start:end], pos_previous_b, v_bi, q_bw))
        t_new_bw.daemon = True
        t_new_bw.start()
        # print q.get()
        vlist_bw.append(q_bw.get())

    pos_new_b = np.vstack(vlist_bw)

    #pos_new_b = generate_next_back(step_no, pos_current_b, pos_previous_b, v_bi,q)
    pos_all_b = np.vstack((pos_all_b, pos_new_b))

    for m in range(len(pos_new_b)):
        pos_new_b[m] = np.hstack((pos_new_b[m][0:10], 1))

    pos_previous_b = pos_current_b
    # pos_previous_b = pos_new_b  #NEED DISCUSSION
    pos_current_b = pos_new_b

    #if v_bi == 1:
        # if abs(z_max[1] - z_max[2]) < 200:
        #if step_no > 100:
            #v_bi = 4
            #tau = 60


    # if step_no < 400:
        # if (step_no + 1) % 5 == 0:
        #    plotter(pos_all,pos_all_b)
        #    pos_saver(pos_all,pos_all_b)

    ########################################################################

#     GrowthRateMean = GrowthRate(pos_current,pos_current_b)
#     z_growth = np.vstack((z_growth, GrowthRateMean))
#     reset_nodes = np.array([0, 0, 0])
#     if np.array_equal(reset_nodes, z_growth[0]):
#         z_growth = np.delete(z_growth, 0, 0)  # save the longest tips of forward and backward
#
# GrowthRateSaver(z_growth) #check!

    ########################################################################

# Print running time
endtime = time.clock()
print('Loop time = ' + str(endtime - starttime) + 's')

#500 cells 10 steps, 0 branch, 30.729136s

#plotter(pos_all,pos_all_b)
#pos_saver(pos_all,pos_all_b)

