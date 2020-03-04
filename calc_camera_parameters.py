'''
Filename: calc_camera_parameters.py
Created Date: Tuesday, February 18th 2020, 6:15:07 pm
Author: Valentin Bruder

Copyright (c) 2020 Visualization Research Institute University of Stuttgart
'''

import sys
import math
import numpy as np
import pandas as pd
import pathlib
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

def rotate_fixed_to(q, look_to, look_from):
    np_cam = look_from - look_to
    np_cam = np.concatenate([[0], np_cam])
    rot_cam = (q * np_cam) * q.conjugate
    # print(rot_cam.vector)
    look_from = look_to - rot_cam.vector
    return look_from

def get_axis(name):
    axis = [0, 0, 0]
    if "x" in name:
        axis[0] = 1
    elif "y" in name:
        axis[1] = 1
    elif "z" in name:
        axis[2] = 1
    elif "-x" in name:
        axis[0] = -1
    elif "-y" in name:
        axis[1] = -1
    elif "-z" in name:
        axis[2] = -1
    return axis

# orbit
def calc_pos_orbit(fovy, bbox_min, bbox_max, num_iterations, step, name):
    data_max = -sys.float_info.max
    data_max = max(abs(bbox_max - bbox_min))

    camera_dist = (data_max * 0.5) / math.tan(math.pi*fovy*0.5/180.)
    cam_initial_pos = np.array([0, 0, bbox_min[2] - camera_dist])
    
    angle = (2.*math.pi / num_iterations) * step
    axis = get_axis(name)

    q = Quaternion(axis=axis, angle=angle)
    data_center = bbox_min + 0.5 * (bbox_max - bbox_min)
    cam_pos = rotate_fixed_to(q, data_center, cam_initial_pos)
    ray = (bbox_min + (bbox_max - bbox_min)*0.5) - cam_pos
    ray /= np.linalg.norm(ray)

    return cam_pos, ray

# diagonal
def calc_pos_diagonal(fovy, bbox_min, bbox_max, num_iterations, step, name):
    begin = bbox_min.copy()
    end = bbox_max.copy()
    # point mirroring according to definition (e.g. "diagonal_zx" is between (0,1,1)->(1,0,0))
    if 'z' in name:
        end, begin = begin, end             # back to front
    if 'x' in name:
        end[0], begin[0] = begin[0], end[0] # right to left
    if 'y' in name:
        end[1], begin[1] = begin[1], end[1] # top to bottom

    # set_look_to(bbox_max)
    diagonal_length = np.linalg.norm(bbox_max - bbox_min)
    camera_dist = (diagonal_length * 0.5) / math.tan(fovy * 0.5 * math.pi / 180.)
    camera_dist *= (1. - step / float(num_iterations))
    cam_pos = bbox_min + (bbox_max - bbox_min)*0.5 + (begin - end)/diagonal_length * camera_dist
    ray = end - cam_pos
    ray /= np.linalg.norm(ray)

    return cam_pos, ray

# path
def calc_pos_path(fovy, bbox_min, bbox_max, num_iterations, step, name):
    step_size = step / float(num_iterations)
    alpha = math.pi*2. * step_size;     # Angular progress on path.
    camera_dist = (bbox_max * 0.5) / math.tan(fovy * math.pi/2. / 180.0)
    cam_pos = bbox_min + 0.5 * (bbox_max - bbox_min)
    dist = 0.0
    data_size = abs(bbox_max - bbox_min)
    ray = np.array([0., 0., 0.])        # View direction of camera.
    up = np.array([0., 1., 0.])         # Camera up vector.
    curve = np.array([0., 0., 0.])      # Curve amplitude.

    if 'x' in name:
        dist = data_size[0]
        cam_dist = 0.5 * max(data_size[1], data_size[2]) / math.tan(fovy*math.pi/2. / 180.)
        cam_pos[0] -= 0.5 * dist + cam_dist
        dist += cam_dist
        ray[0] = 1.
        curve[2] = data_size[2]
    elif 'y' in name:
        dist = data_size[1]
        cam_dist = 0.5 * max(data_size[0], data_size[2]) / math.tan(fovy*math.pi/2. / 180.)
        cam_pos[1] -= 0.5 * dist + cam_dist
        dist += cam_dist
        ray[1] = 1.
        curve[2] = data_size[2]
        up[0] = 1.
        up[1] = 0.
    elif 'z' in name:
        dist = data_size[2]
        cam_dist = 0.5 * max(data_size[0], data_size[1]) / math.tan(fovy*math.pi/2. / 180.)
        cam_pos[2] -= 0.5 * dist + cam_dist
        dist += cam_dist
        curve[0] = data_size[0]
        ray[2] = 1.
    # curves
    if "sin" in name:
        curve *= 0.25
        a = math.sin(alpha)
        da = math.cos(alpha)
        cam_pos += a * curve
        cam_pos += (step_size * dist) * ray
        da /= max(curve)
        q = Quaternion(axis=up, angle=da)
        ray = ((q * np.concatenate([[0], ray])) * q.conjugate).vector
        ray /= np.linalg.norm(ray)
    # TODO: add cos 
    else:
        cam_pos += (step_size*dist) * ray
    return cam_pos, ray


def calc_pos(fovy, bbox_min, bbox_max, num_iterations, step, name):
    cam_pos = np.array([ 0,  0,  0])
    if "orbit" in name:
        cam_pos, ray = calc_pos_orbit(fovy, bbox_min, bbox_max, num_iterations, step, name)
    elif "diagonal" in name:
        cam_pos, ray = calc_pos_diagonal(fovy, bbox_min, bbox_max, num_iterations, step, name)
    elif "path" in name:
        cam_pos, ray = calc_pos_path(fovy, bbox_min, bbox_max, num_iterations, step, name)
    return cam_pos, ray

''' 
Fit a plane to a set of points in 3D (minimize the distances).
@see https://stackoverflow.com/questions/35070178/fit-plane-to-a-set-of-points-in-3d-scipy-optimize-minimize-vs-scipy-linalg-lsts 
'''
def get_plane(coords):
    # barycenter of the points
    # compute centered coordinates
    G = coords.sum(axis=0) / coords.shape[0]

    # run SVD
    u, s, vh = np.linalg.svd(coords - G)

    # unitary normal vector
    u_norm = vh[2, :]
    return u_norm

'''
plot the plane from 
@see https://stackoverflow.com/questions/3461869/plot-a-plane-based-on-a-normal-vector-and-a-point-in-matlab-or-matplotlib
'''
from mpl_toolkits.mplot3d import axes3d
from matplotlib.patches import Circle, PathPatch
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import art3d

def rotation_matrix(d):
    sin_angle = np.linalg.norm(d)
    if sin_angle == 0:return np.identity(3)
    d /= sin_angle
    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z, normal):
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised
    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector    
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    pathpatch._segment3d += delta


def plot_plane(ax, point, normal, size=10, color='y'):    
    p = Circle((0, 0), size, facecolor = color, alpha = .2)
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z=0, normal=normal)
    pathpatch_translate(p, (point[0], point[1], point[2]))


def project_2d(coords, normal, rays):
    # some point on the plane
    point = coords[int(len(coords)/2)]
    # find d via plane equation
    d = -normal[0]*point[0] - normal[1]*point[1] - normal[2]*point[2]

    # calculate projected coords (perpendicular onto the plane) 
    coords_proj = []
    rays_proj = []
    for p, r in zip(coords, rays):
        # project point using perpendicular distance to plane
        dist = normal.dot(p) + d
        coords_proj.append(p - dist*normal)
        # project ray end point
        dist_ray = normal.dot(r - p) + d
        # ray goes from coords_proj towards rays_proj
        rays_proj.append((r - p) - dist_ray*normal)

    # change of basis to 2d
    # the plane normal is our Z
    # we need at least 3 points
    assert(len(coords_proj) >= 3)
    u = coords_proj[1] - coords_proj[0]
    X = u - u.dot(normal)*normal
    Y = np.cross(normal, X)

    # transform all points to 2d coord system
    coords_trans = []
    rays_trans = []
    coords_trans_x = []
    coords_trans_y = []
    rays_trans_x = []
    rays_trans_y =[]

    for p, r in zip(coords_proj, rays_proj):
        coords_trans.append(np.array([p.dot(X), p.dot(Y)]))
        coords_trans_x.append(p.dot(X))
        coords_trans_y.append(p.dot(Y))
        rays_trans.append(np.array([r.dot(X), r.dot(Y)]))
        rays_trans_x.append(r.dot(X))
        rays_trans_y.append(r.dot(Y))

    print('coords transformed to 2d: ' + str(coords_trans))

    ### plot 2d projection
    fig = plt.figure()
    plt.plot(coords_trans_x, coords_trans_y, '-o')
    plt.quiver(coords_trans_x, coords_trans_y, rays_trans_x, rays_trans_y, color=[0,0.7,0,0.4])
    plt.show()

    return coords_trans, rays_trans

'''
draw unit cube 
TODO: adapt to actual bounding box 
'''
def draw_cube(ax):
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r,r,r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="red")


'''
Calculate and plot positions for the camera paths used during the TRRojan measurements.
'''
def trrojan_paths(num_iterations):
    fovy = 60   # always 60 degrees for the point data benchmark
    bbox_min = np.array([-1, -1, -1])   # TODO: depends on the data set
    bbox_max = np.array([ 1,  1,  1])   # TODO: depends on the data set
    camera_path_names = ["diagonal_x", "diagonal_y", "diagonal_z", 
                         "orbit_x", "orbit_y", 
                         "path_x", "path_y", "path_z", 
                         "path_sin_x", "path_sin_y", "path_sin_z"
                        ]

    for name in camera_path_names:
        pos_x = []
        pos_y = []
        pos_z = []
        ray_x = []
        ray_y = []
        ray_z = []

        coords = []
        rays = []
        
        print('~~~~~ ' + name)
        for i in range(num_iterations):
            pos, ray = calc_pos(fovy, bbox_min, bbox_max, num_iterations, i, name)
            
            pos_x.append(pos[0])
            pos_y.append(pos[1])
            pos_z.append(pos[2])
            coords.append(pos)

            ray_x.append(ray[0])
            ray_y.append(ray[1])
            ray_z.append(ray[2])
            rays.append(ray)
            # print('pos: ' + str(pos))
            # print('dir: ' + str(ray))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # camera positions
        ax.plot(pos_x, pos_y, pos_z, '-o')
        # view direction as arrows 
        ax.quiver(pos_x, pos_y, pos_z, ray_x, ray_y, ray_z, color=[0,0.7,0,0.4])
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])

        # calculate projection plane
        normal = get_plane(np.array(coords))
        print('normal: ' + str(normal))
        # point on plane
        point = coords[int(len(coords)/2)]

        ## plot 3d points
        # plot the plane
        plot_plane(ax, point, normal, size=4) 
        # plot bounding box (unit cube)
        draw_cube(ax)
        plt.title(name)
        plt.show()

        # project points to 2d and plot in 2d
        coords_trans, rays_trans = project_2d(coords, normal, rays)
        write2csv(coords_trans, rays_trans, 11, 10, name)

'''
'''
def write2csv(coords, rays, cam_step_count, cam_step_stride, name):
    coords_trans_x = []
    coords_trans_y = []
    rays_trans_x = []
    rays_trans_y = []
    for c,r in zip(coords, rays):
        coords_trans_x.append(c[0])
        coords_trans_y.append(c[1])
        rays_trans_x.append(r[0])
        rays_trans_y.append(r[1])    

    cam_steps = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    if len(cam_steps) != len(coords):
        cam_steps = []
        for i in range(0, cam_step_count, cam_step_stride):
            cam_steps.append(i)

    df = pd.DataFrame()
    df["pos_x"] = coords_trans_x
    df["pos_y"] = coords_trans_y
    df["view_x"] = rays_trans_x
    df["view_y"] = rays_trans_y    
    df["camera_step"] = cam_steps
    df.set_index("camera_step", inplace=True)
    export_dir = pathlib.Path("camera_prototype", "public", name)
    pathlib.Path(export_dir).mkdir(parents=True)
    df.to_csv(pathlib.Path(export_dir, "points.csv"))
    print(df)

'''
driver code
'''
def main():
    if len(sys.argv) == 1:
        trrojan_paths(11)
    if len(sys.argv) == 2:
        trrojan_paths(int(sys.argv[1]))
    elif len(sys.argv) != 4:
        print("Number arguments must be either 0, 1 or 3:")
        print("# camera iterations (1), rotations file name (2), translation file name (3)")
        return 

    ### recorded camera path
    stride = int(sys.argv[1])   # sampling distance between recorded points
    rot_file_name = ""
    trans_file_name = ""

    # read file name of rotations
    if len(sys.argv) > 2:
        rot_file_name = str(sys.argv[2])
    file_rot = open(rot_file_name, 'r')
    rot = []
    for l in file_rot:
        for s in l.split(';'):
            vals = s.split(' ')
            if len(vals) == 5:
                vals.pop(0)
            if len(vals) != 4:
                break
            q = Quaternion(vals[0], vals[1], vals[2], vals[3])
            rot.append(q)
    file_rot.close()

    # read file name of translations
    if len(sys.argv) > 3:
        trans_file_name = str(sys.argv[3])
    file_trans = open(trans_file_name, 'r')
    trans = []
    for l in file_trans:
        for s in l.split(';'):
            vals = s.split(' ')
            if len(vals) == 4:
                vals.pop(0)
            if len(vals) != 3:
                break
            t = np.array([float(vals[0]), float(vals[1]), float(vals[2])])
            trans.append(t)
    file_trans.close()

    pos_x = []
    pos_y = []
    pos_z = []
    coords = []
    rays = []
    # calculate every n-th position
    for i,t in enumerate(trans):
        if i % stride != 0:
            continue
        q = rot[i]
        pos = q.rotate(t)
        coords.append(pos)
        pos_x.append(pos[0])
        pos_y.append(pos[1])
        pos_z.append(pos[2])
        # TODO: cam is always facing the center in the custom path for now
        rays.append(-pos / np.linalg.norm(-pos))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plot camera positions
    ax.plot(pos_x, pos_y, pos_z, '-o')

    # calculate projection plane
    # TODO: validate plane, this does not look correct for the custom cam.
    normal = get_plane(np.array(coords))
    print('normal: ' + str(normal))
    # point on plane
    point = coords[int(len(coords)/2)]

    ## plot 3d points
    # plot the plane
    plot_plane(ax, point, normal, size=4) 
    # plot bounding box (unit cube)
    draw_cube(ax)

    plt.title("Custom camera path")
    plt.show()
    
    # project points to 2d and plot in 2d
    coords_trans, rays_trans = project_2d(coords, normal, rays)
    write2csv(coords_trans, rays_trans, len(trans), stride, "custom")

main()
