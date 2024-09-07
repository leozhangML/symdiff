import os
print(os.getcwd())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qm9.data.prepare.download import prepare_dataset

#datadir = "/data/ziz/not-backed-up/datasets-ziz-all/qm9"
#dataset = "qm9"
#_ = prepare_dataset(datadir, dataset, subset=None, splits=None, cleanup=True, force_download=True)

from mpl_toolkits.mplot3d import Axes3D

black = (0, 0, 0)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_aspect('auto')
#ax.view_init(elev=camera_elev, azim=camera_azim)
ax.set_facecolor(black)
# ax.xaxis.pane.set_edgecolor('#D0D0D0')
ax.xaxis.pane.set_alpha(0)
ax.yaxis.pane.set_alpha(0)
ax.zaxis.pane.set_alpha(0)
ax._axis3don = False

ax.w_xaxis.line.set_color("black")
