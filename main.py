import glob
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from loader import ObjLoader, BinvoxLoader


obj_files = glob.glob("data/*.obj")
white = np.array([255, 255, 255])


for i, obj_file in enumerate(obj_files):
    vox = BinvoxLoader(obj_file.replace('.obj', '.binvox'))
    obj = ObjLoader(obj_file)
    obj.rescale(vox.dim - 1)

    nbrs = NearestNeighbors(n_neighbors=2).fit(obj.points)
    distances, indices = nbrs.kneighbors(vox.points)

    directory = os.path.split(obj_file)[0]
    data = np.zeros([vox.dim, vox.dim, vox.dim, 4], dtype=np.uint8)

    for vi, oi in enumerate(indices):
        x = vox.indices[0][vi]
        y = vox.indices[1][vi]
        z = vox.indices[2][vi]

        mtltags = obj.mtltags[oi]
        prev_dist = -1

        for k, m in enumerate(mtltags):
            material = obj.mtl.get(m)
            color = material.get('Kd')
            dist = np.linalg.norm(white - np.array(color))

            if dist > prev_dist:
                data[x, y, z, 1:] = color
                prev_dist = dist

            # TODO: sample from texture images
            # if 'map_Kd' in material:
            #     a, b, c = obj.faces[k][1]
            #     a = [obj.texcoords[a]]
            #     b = [obj.texcoords[b]]
            #     c = [obj.texcoords[c]]
            #     centroid = np.concatenate([a, b, c]).mean(axis=0)
            #     image = material.get('map_Kd')
            #     h, w, _ = image.shape
            #     centroid *= [w, h]
            #     u, v = np.int32(centroid)
            #     if u > w - 1:
            #         u = int(u - w * np.ceil(float(u) / w))
            #     elif u < 0:
            #         u = int(u + w * np.ceil(float(abs(u)) / w))
            #     if v > h - 1:
            #         v = int(v - h * np.ceil(float(v) / h))
            #     elif v < 0:
            #         v = int(v + h * np.ceil(float(abs(v)) / h))
            #     data[x, y, z, 1:] = image[v, u]

    data[:, :, :, 0] = vox.data
    np.save(obj_file.replace('.obj', '.npy'), data)
    print(i, obj_file)
