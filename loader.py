import numpy as np
import os
import cv2
import binvox


class ObjLoader(object):

    def __init__(self, filename):
        self.vertices = []
        self.faces = []
        self.texcoords = []
        self.points = []
        self.mtltags = []
        directory = os.path.split(filename)[0]
        mtltag = None

        for line in open(filename, 'r'):
            if line.startswith('#'): continue
            if line.startswith('s'): continue
            if line.startswith('o'): continue

            values = line.split()
            key = values[0]

            if key == 'mtllib':
                mtl_path = os.path.join(directory, values[1])
                self.mtl = MTL(mtl_path)
            elif key in ('usemtl', 'usemat'):
                mtltag = values[1]
            elif key == 'vt':
                self.texcoords.append(map(np.float32, values[1:3]))
            elif key == 'v':
                v = list(map(np.float32, values[1:4]))
                self.vertices.append(v)
            elif key == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    f = np.int32(w[0]) - 1
                    t = np.int32(w[1]) - 1 if len(w) >= 2 and len(w[1]) > 0 else -1
                    n = np.int32(w[2]) - 1 if len(w) >= 3 and len(w[2]) > 0 else -1
                    face.append(f)
                    texcoords.append(t)
                    norms.append(n)
                self.faces.append((face, texcoords, norms, mtltag))

        self.vertices = np.array(self.vertices)
        self.texcoords = np.array(self.texcoords)

        for face in self.faces:
            f = face[0]
            t = face[3]
            a = self.vertices[f[0]]
            b = self.vertices[f[1]]
            c = self.vertices[f[2]]

            centroid = (a + b + c) / 3.
            self.points.append(centroid)

            p1 = (a + b + centroid) / 3.
            p2 = (a + c + centroid) / 3.
            p3 = (b + c + centroid) / 3.
            self.points.append(p1)
            self.points.append(p2)
            self.points.append(p3)

            for i in range(4):
                self.mtltags.append(t)

        self.points = np.array(self.points)
        self.mtltags = np.array(self.mtltags)

        point_max = self.points.max(axis=0)
        point_min = self.points.min(axis=0)
        self.bbox = point_max - point_min

        translate = 0.5 * self.bbox - point_max
        self.vertices += translate
        self.points += translate

    def rescale(self, scale):
        scale /= self.bbox.max()
        self.vertices *= scale
        self.points *= scale
        self.bbox *= scale


class MTL(object):

    def __init__(self, filename):
        self.contents = {}
        directory = os.path.split(filename)[0]

        if not os.path.exists(filename):
            return None

        for line in open(filename, 'r'):
            if line.startswith('#'): continue

            values = line.split()
            if not values: continue
            key = values[0]

            if key == 'newmtl':
                mtl = self.contents[values[1]] = {}
                continue
            elif mtl is None:
                raise(ValueError, "mtl file doesn't start with newmtl")

            if key == 'map_Kd':
                texpath = values[1].replace("./", "", 1)
                texpath = os.path.join(directory, texpath)
                if os.path.isfile(texpath):
                    img = cv2.imread(texpath, True)
                    if img is not None:
                        assert img.dtype == np.uint8
                        mtl[key] = img
            elif key == 'Kd':
                kd = map(np.float32, values[1:])
                kd = map(lambda n: np.uint8(255 * n), kd)
                mtl[key] = list(kd)
            elif len(values[1:]) > 1:
                mtl[key] = values[1:]
            else:
                mtl[key] = values[1]

    def __getitem__(self, key):
        return self.contents[key]

    def get(self, key, default=None):
        return self.contents.get(key, default)


class BinvoxLoader(object):

    def __init__(self, filename):
        with open(filename, 'rb') as f:
            model = binvox.read_as_3d_array(f)
            self.data = model.data.astype(np.uint8)

        self.points = []
        self.dim = model.dims[0]
        self.indices = np.where(self.data == 1)
        total = self.data.sum()
        translate = (self.dim - 1) * 0.5

        for i in range(total):
            x = self.indices[0][i] - translate
            y = self.indices[1][i] - translate
            z = self.indices[2][i] - translate
            point = [x, y, z]
            self.points.append(point)
