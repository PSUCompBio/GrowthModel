# This file is used to write vtk format for Bidirectional Growth
# Remember to change the input text files name

import math


cellnumber = []         # Cell number: saved in column 6
stepnumber = []         # Step number: saved in column 7
previousnode = []       # Previous node number: saved in column 8
currentnode = []        # Current node number: saved in column 9

r_fwd = []              # r coordinates: saved in column 0
theta_fwd = []          # theta coordinates: saved in column 1
z_fwd = []              # z coordinates: saved in column 2

uniquenumber = []
uniquenumber1 = []

x_fwd = []              # x coordinates: r * cos(theta)
y_fwd = []              # y coordinates: r * sin(theta)

new_row = []

# Open the output from Growth model and save in a list; Remember to change the file's name
with open('pos_bi_fwd_500_20.txt') as f:
    content = f.readlines()

# Read all useful information from the list
for line in content:
    currentnode.append(line.split()[9])
    previousnode.append(line.split()[8])
    cellnumber.append(line.split()[6])
    uniquenumber.append(line.split()[6] + '-' + line.split()[9])       # Create unique numbers for cell number and current node
    uniquenumber1.append(line.split()[6] + '-' + line.split()[8])      # Create unique numbers for cell number and previous node
    stepnumber.append(line.split()[7])

    r_fwd.append(line.split()[0])
    theta_fwd.append(line.split()[1])
    z_fwd.append(line.split()[2])

x_fwd = [float(i) * math.cos(float(j)) for i in r_fwd for j in theta_fwd]
y_fwd = [float(i) * math.sin(float(j)) for i in r_fwd for j in theta_fwd]
z_fwd = [float(i) for i in z_fwd]


for i_fwd in uniquenumber1:
    new_row.append(uniquenumber.index(i_fwd))

# # Write the connection relation to text file
# with open('result.txt', 'w') as f:
#     for i in range(len(new_row)):
#         f.write("%s %s \n" % (str(i), str(new_row[i])))

####################### Deal with backward growth ###############################
cellnumber_bw = []
stepnumber_bw = []
currentnode_bw = []
previousnode_bw = []

r_bw = []
theta_bw = []
z_bw = []

uniquenumber_bw = []
uniquenumber1_bw = []

x_bw = []
y_bw = []

new_row_bw = []

with open('pos_bi_bw_500_20.txt') as f:                 # Remember to change the file name
    content_bw = f.readlines()
for line in content_bw:
    currentnode_bw.append(line.split()[9])
    previousnode_bw.append(line.split()[8])
    cellnumber_bw.append(line.split()[6])
    uniquenumber_bw.append(line.split()[6] + '-' + line.split()[9])
    uniquenumber1_bw.append(line.split()[6] + '-' + line.split()[8])
    stepnumber_bw.append(line.split()[7])

    r_bw.append(line.split()[0])
    theta_bw.append(line.split()[1])
    z_bw.append(line.split()[2])

x_bw = [float(i) * math.cos(float(j)) for i in r_bw for j in theta_bw]
y_bw = [float(i) * math.sin(float(j)) for i in r_bw for j in theta_bw]
z_bw = [float(i) for i in z_bw]

for i_bw in uniquenumber1_bw:
    new_row_bw.append(uniquenumber_bw.index(i_bw))

# # # Write the connection relation to text file
# # with open('result.txt', 'w') as f:
# #     for i in range(len(new_row)):
# #         f.write("%s %s \n" % (str(i), str(new_row[i])))


####################### Start to write VTK format ############################
stepnumber = [int(float(i)) for i in stepnumber]
cellnumber = [int(float(i)) for i in cellnumber]


NoStep = max(stepnumber)
NoCell = max(cellnumber)
CellType = 3                    # In VTK format, Type 3 is VTK_LINE

counter = 0

while counter < NoStep:
# while counter < 2:

    new_x = []
    new_xb = []
    new_yb = []
    new_y = []
    new_z = []
    new_zb = []
    new_connection = []
    new_connectionb = []

    for i,s in enumerate(stepnumber):

        if s <= counter:
            new_x.append(x_fwd[i])
            new_xb .append(x_bw[i])
            new_y.append(y_fwd[i])
            new_yb.append(y_bw[i])
            new_z.append(z_fwd[i])
            new_zb.append(z_bw[i])
            new_connection.append(new_row[i])
            new_connectionb.append(new_row_bw[i])
        else:
            totalx = new_x + new_xb
            totaly = new_y + new_yb
            totalz = new_z + new_zb
            totalconnection = new_connection + [r + len(new_connection) for r in new_connectionb]
            with open('step%s.vtk' %s,'wb') as output:
                output.write("# vtk DataFile Version 3.0\n")
                output.write("%i Cells\n" %(NoCell+1))
                output.write("ASCII\n")
                output.write("DATASET UNSTRUCTURED_GRID\n")
                output.write("POINTS %d FLOAT\n" % (len(totalx)))
                for i in range(len(totalx)):
                    output.write("%s %s %s\n" % (str(totalx[i]), str(totaly[i]),str(totalz[i])))
                output.write(("CELLS %d %d\n" % (len(totalconnection), len(totalconnection) * 3)))
                for j in range(len(totalconnection)):
                    output.write("%d %s %s\n" % (2, str(j), str(totalconnection[j])))
                output.write("CELL_TYPES %d\n" % (len(totalconnection)))
                for m in range(len(totalconnection)):
                    output.write("%d\n" % CellType)
                output.write("CELL_DATA %d\n" % (len(totalx)))
                output.write("SCALARS cell_scalars int 1\nLOOKUP_TABLE default\n")
                for n in range(len(new_x)):
                    output.write("0\n")
                for p in range(len(new_xb)):
                    output.write("1\n")

            counter +=1
            break
