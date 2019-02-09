import argparse
import vtk
import glob
import os
import numpy as np


def create_voxel():
    numberOfVertices = 8

    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(0, 1, 0)
    points.InsertNextPoint(1, 1, 0)
    points.InsertNextPoint(0, 0, 1)
    points.InsertNextPoint(1, 0, 1)
    points.InsertNextPoint(0, 1, 1)
    points.InsertNextPoint(1, 1, 1)

    voxel = vtk.vtkVoxel()
    for i in range(0, numberOfVertices):
        voxel.GetPointIds().SetId(i, i)

    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(points)
    ug.InsertNextCell(voxel.GetCellType(), voxel.GetPointIds())

    gf = vtk.vtkGeometryFilter()
    gf.SetInputData(ug)
    gf.Update()
    return gf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_file', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data = np.load(args.npy_file)
    indices = np.where(data[:, :, :, 0] == 1)
    n_voxels = indices[0].shape[0]

    points = vtk.vtkPoints()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetName("colors")
    colors.SetNumberOfComponents(4)

    for i in range(n_voxels):
        x = indices[0][i]
        y = indices[1][i]
        z = indices[2][i]
        r = data[x, y, z, 1]
        g = data[x, y, z, 2]
        b = data[x, y, z, 3]
        colors.InsertTuple4(i, r, g, b, 128)
        points.InsertNextPoint(x, y, z)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().SetScalars(colors)

    # create cube
    cube = create_voxel()
    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetColorModeToColorByScalar()
    glyph3D.SetSourceData(cube.GetOutput())
    glyph3D.SetInputData(polydata)
    glyph3D.ScalingOff()
    glyph3D.Update()

    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(glyph3D.GetOutput())

    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetAmbient(0.2)
    actor.RotateX(15)
    actor.RotateY(-45)

    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(400, 400)

    # create a renderwindowinteractor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renWin)

    # assign actor to the renderer
    ren.AddActor(actor)
    ren.SetBackground(0, 0, 0)

    # enable user interface interactor
    interactor.Initialize()
    interactor.Start()
    renWin.Render()

    # TODO: screenshot
    # head, tail = os.path.split(args.npy_file)
    # name, ext = os.path.splitext(tail)
    # w2if = vtk.vtkWindowToImageFilter()
    # w2if.SetInputData(renWin)
    # w2if.SetMagnification(3)
    # w2if.Update()
    # writer = vtk.vtkPNGWriter()
    # writer.SetFileName("data/{0}.png".format(name))
    # writer.SetInputData(w2if.GetOutput())
    # writer.Write()
