import os
import vtk
import numpy as np


##############################################################################
# General visualization utilities
##############################################################################
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import seaborn as sns


class Color:
    White = (255, 255, 255)
    Black = (0, 0, 0)
    Gray = (179, 179, 179)
    DarkGray = (60, 60, 60)
    LightGray = (220, 220, 220)
    Green = (0, 120, 0)
    DarkGreen = (0, 51, 0)
    LightGreen = (0, 255, 0)
    Orange = (255, 156, 28)
    Blue = (0, 102, 255)
    Purple = (255, 0, 255)
    Yellow = (255, 255, 0)
    Red = (255, 0, 0)


##############################################################################
# VTK visualization utilities
##############################################################################

class VtkPointCloud:
    """
  Visualizes a point cloud (colored by its labels) using vtk.
  """

    def __init__(self, points, gt_points=[], pred_points=[], point_size=1.0, use_rgb=False, color=Color.White):
        """
    points: (D,)
    gt_points (optional): (D,) binary values (0 or 1) where 1 means it belongs to object
    pred_points (optional): (D,) binary values (0 or 1) where 1 means it is predicted
    """
        if len(gt_points) > 0: assert (len(points) == len(gt_points))
        if len(pred_points) > 0: assert (len(points) == len(pred_points))

        self.vtk_poly_data = vtk.vtkPolyData()
        self.vtk_points = vtk.vtkPoints()
        self.vtk_cells = vtk.vtkCellArray()
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)

        self.vtk_poly_data.SetPoints(self.vtk_points)
        self.vtk_poly_data.SetVerts(self.vtk_cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtk_poly_data)
        self.vtk_actor = vtk.vtkActor()
        self.vtk_actor.SetMapper(mapper)
        self.vtk_actor.GetProperty().SetPointSize(point_size)

        # Create point cloud
        for i, point in enumerate(points):
            if use_rgb:
                rgb = np.array(point[3:6], dtype=np.int8)

                self.add_point(point, color=rgb)
                continue

            if len(gt_points) > 0 and len(pred_points) > 0:
                # If there is ground truth + prediction points
                if gt_points[i] == 1 and pred_points[i] == 1:
                    # True positive
                    self.add_point(point, color=Color.LightGreen)
                elif gt_points[i] == 0 and pred_points[i] == 0:
                    # True negative
                    self.add_point(point, color=color)
                elif gt_points[i] == 0 and pred_points[i] == 1:
                    # False positive
                    self.add_point(point, color=Color.Red)
                elif gt_points[i] == 1 and pred_points[i] == 0:
                    # False negative
                    self.add_point(point, color=Color.Yellow)
                else:
                    raise Exception('Should not have such a situation')

            elif len(gt_points) > 0:
                # If there is only ground truth points
                if gt_points[i] == 1:
                    self.add_point(point, color=Color.LightGreen)
                else:
                    self.add_point(point, color=color)

            elif len(pred_points) > 0:
                # If there is only ground truth points
                if pred_points[i] == 1:
                    self.add_point(point, color=Color.Orange)
                else:
                    self.add_point(point, color=color)

            else:
                self.add_point(point, color=color)

    def add_point(self, point_with_label, color=Color.White):
        pointId = self.vtk_points.InsertNextPoint(point_with_label[0:3])
        self.vtk_cells.InsertNextCell(1)
        self.vtk_cells.InsertCellPoint(pointId)

        self.colors.InsertNextTuple3(color[0], color[1], color[2])
        self.vtk_poly_data.GetPointData().SetScalars(self.colors)
        self.vtk_poly_data.Modified()

        self.vtk_cells.Modified()
        self.vtk_points.Modified()


def vtk_box_3D(points, line_width=1, color=Color.LightGreen):
    """
  3D bbox for display on the vtk visualization.
  """
    # Create a vtkPoints object and store the points in it
    vtk_points = vtk.vtkPoints()
    for pt in points:
        vtk_points.InsertNextPoint(pt)

    # Setup the colors array
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    lineIds = [(0, 1), (0, 3), (1, 2), (2, 3),  # Lines in the bottom half
               (4, 5), (4, 7), (5, 6), (6, 7),  # Lines in the top half
               (0, 4), (1, 5), (2, 6), (3, 7)]  # Lines from bottom to top
    lines = vtk.vtkCellArray()
    for lineId in lineIds:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, lineId[0])
        line.GetPointIds().SetId(1, lineId[1])
        lines.InsertNextCell(line)
        colors.InsertNextTuple(color)

    # Create a polydata to store everything in
    lines_poly_data = vtk.vtkPolyData()
    lines_poly_data.SetPoints(vtk_points)
    lines_poly_data.SetLines(lines)
    lines_poly_data.GetCellData().SetScalars(colors)

    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(lines_poly_data)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(line_width)
    return actor


class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, ren_win_interactor, ren, ren_win, dict_key_to_actors_to_hide, scene_name, ss_dir='vis_tmp'):
        """
    dict_key_to_actors_to_hide is a dictionary with a 'Keyboard key' to List of Vtk Actors mapping
    where if the particular keyboard key has been pressed, the corresponding Vtk Actors' visibility
    will be toggled.
    Can also be used to take screenshots into ss_dir by pressing 's'.
    """
        if not os.path.exists(ss_dir): os.mkdir(ss_dir)
        self.ss_dir = ss_dir
        self.ren = ren
        self.ren_win = ren_win
        self.ren_win_interactor = ren_win_interactor
        self.scene_name = scene_name
        self.dict_key_to_actors_to_hide = dict_key_to_actors_to_hide
        self.AddObserver("KeyPressEvent", self.key_press_event)

    def key_press_event(self, obj, event):
        pressed_key = self.ren_win_interactor.GetKeySym()

        if pressed_key == 's':
            cam = self.ren.GetActiveCamera()
            # Choose the highest count of the
            paths = [path for path in os.listdir(self.ss_dir) if path.split('_')[0] == 'screenshot']
            ss_count = max([int(path.split('_')[-1]) for path in paths]) + 1 if len(paths) > 0 else 1
            ss_name = '%s_screenshot_%03d' % (self.scene_name, ss_count)
            print('\n--- Screenshot %s ---' % ss_name)
            print('pos = ' + str(cam.GetPosition()))
            print('fp = ' + str(cam.GetFocalPoint()))

            # Screenshot code
            screenshot_name = os.path.join(self.ss_dir, ss_name)
            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(self.ren_win)
            w2if.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(screenshot_name)
            writer.SetInputData(w2if.GetOutput())
            writer.Write()

            return

        for key, actors in self.dict_key_to_actors_to_hide.items():
            if pressed_key == key:
                for actor in actors:
                    visibility = actor.GetVisibility()
                    actor.SetVisibility(1 - visibility)
                self.refresh()

    def refresh(self, resetCamera=False):
        if resetCamera:
            self.ren.ResetCamera()
        self.ren_win.Render()
        self.ren_win_interactor.Start()


def start_render(vtkActors, key_to_actors_to_hide={}, window_wh=(300, 300), background_col=(0, 0, 0),
                 background_col2=None,
                 camera=None, scene_name=''):
    """
  Start rendering a vtk actor.

  eyToActorsToHide is a dictionary with a 'Keyboard key' to List of Vtk Actors mapping
  where if the particular keyboard key has been pressed, the corresponding Vtk Actors' visibility
  will be toggled.

  Example:
    `start_render([vtkPC.vtkActor])`: Will render a point cloud without having any keypress events.
    `start_render([], { 'h': [vtkPC.vtkActor] })`: Will render a point cloud which will toggle its
      visibility if 'h' has been pressed.
  """
    # Renderer
    renderer = vtk.vtkRenderer()
    for vtkActor in vtkActors:
        renderer.AddActor(vtkActor)
    for key, actors in key_to_actors_to_hide.items():
        for actor in actors:
            renderer.AddActor(actor)

    background_col = (col / 255. for col in background_col)
    renderer.SetBackground(*background_col)
    if background_col2 is not None:
        renderer.GradientBackgroundOn()
        background_col2 = (col / 255. for col in background_col2)
        renderer.SetBackground2(*background_col2)

    if camera is not None:
        renderer.SetActiveCamera(camera)
    else:
        renderer.ResetCamera()

    # Setup render window, renderer, and interactor
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(*window_wh)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetInteractorStyle(MyInteractorStyle(render_window_interactor, renderer, render_window,
                                                                  key_to_actors_to_hide, scene_name))
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.Start()
    render_window.Render()


def visualise_features(feats, ys, filename="assets/tsne_plot"):
    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(feats, ys)
    fig = scatter(tsne_result, ys, filename)
    return fig


def scatter(x, colors, filename="assets/tsne_plot"):
    # choose a color palette with seaborn.
    num_classes = np.max(colors) + 1
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.savefig(filename + ".png")
    plt.close()
    return f