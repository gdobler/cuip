
import ternary
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# -- Load a night to plot.
lc.loadnight(pd.datetime(2014, 6, 16), True, False)

# -- Calculate the median color for each source, and find the proportion of each
# -- channel.
mlc = np.median(lc.lcs, axis=0)
mlc_sum = np.sum(mlc, axis=1).reshape(-1, 1)
mlc_per = mlc / mlc_sum
# -- Create figure (median colors plot).
fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(11, 6), facecolor="w")
# -- Define plotting constants.
fontsize = 16
tick_formats = "%.1f"
axes_colors = {'b': 'r', 'r': 'g', 'l': 'b'}
# -- Create ternary axes.
tax1 = ternary.TernaryAxesSubplot(ax=ax1, scale=100)
tax2 = ternary.TernaryAxesSubplot(ax=ax2, scale=30)
# -- Title.
fig.suptitle("Median Color for Each Source ({})".format(lc.night.date()),
    fontsize=fontsize)
tax1.set_title("Entire range")
tax2.set_title("Zoomed region")
# -- Format boundaries, etc.
for ax, mult in [[tax1, 10], [tax2, 5]]:
    ax.gridlines(color="black", multiple=mult, linewidth=0.5, ls='-')
    ax.boundary(linewidth=2.0, axes_colors=axes_colors)
    ax.ax.axis("equal")
    ax.ax.axis("off")
    ax.left_axis_label("Blue (%)", fontsize=fontsize, offset=0.17, color="b")
    ax.right_axis_label("Green (%)", fontsize=fontsize, offset=0.17, color="g")
    ax.bottom_axis_label("Red (%)", fontsize=fontsize, offset=0.03, color="r")
# -- Plot the entire range.
tax1.ticks(multiple=10, offset=0.02, axes_colors=axes_colors)
tax1.line((25, 25, 0), (65, 25, 0), color='orange', lw=2.0)
tax1.line((25, 25, 0), (25, 65, 0), color='orange', lw=2.0)
tax1.line((25, 65, 0), (65, 25, 0), color='orange', lw=2.0)
tax1.scatter(mlc_per * 100, c=mlc_per, alpha=0.7, s=40, label="LightCurve")
# -- Plot the zoomed region.
tax2.set_axis_limits({'b': [25, 65], 'l': [10, 50], 'r': [25, 65]})
tax2.get_ticks_from_axis_limits(multiple=5)
tax2.set_custom_ticks(fontsize=10, offset=0.025, multiple=5,
    axes_colors=axes_colors, tick_formats=tick_formats)
points_c = tax2.convert_coordinates(mlc_per * 100, axisorder='brl')
tax2.scatter(points_c, c=mlc_per, s=40)
# -- Reset position of ternary plots and show.
tax1.ax.set_position([0.01, 0.05, 0.46, 0.8])
tax2.ax.set_position([0.50, 0.05, 0.46, 0.8])
tax1.resize_drawing_canvas()
tax2.resize_drawing_canvas()
ternary.plt.show()


# -- Color by bigoff time.
bigoffs = lc.lc_bigoffs.loc[lc.night]
bigoffs = bigoffs[~bigoffs.isnull()]
idx = np.array(bigoffs.index).astype(int)
colors = MinMaxScaler().fit_transform(bigoffs.values.reshape(-1, 1)) \
    .reshape(1, -1)[0]
# -- Create figure (median colors plot).
fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, figsize=(11, 6),
    gridspec_kw={"width_ratios": [5.4, 5.4, 0.2]})
# -- Define plotting constants.
fontsize = 16
tick_formats = "%.1f"
axes_colors = {'b': 'r', 'r': 'g', 'l': 'b'}
# -- Create ternary axes.
tax1 = ternary.TernaryAxesSubplot(ax=ax1, scale=100)
tax2 = ternary.TernaryAxesSubplot(ax=ax2, scale=30)
# -- Title.
fig.suptitle("Bigoff Median Color for Each Source ({})".format(lc.night.date()),
    fontsize=fontsize)
tax1.set_title("Entire range")
tax2.set_title("Zoomed region")
# -- Format boundaries, etc.
for ax, mult in [[tax1, 10], [tax2, 5]]:
    ax.gridlines(color="black", multiple=mult, linewidth=0.5, ls='-')
    ax.boundary(linewidth=2.0, axes_colors=axes_colors)
    ax.ax.axis("equal")
    ax.ax.axis("off")
    ax.left_axis_label("Blue (%)", fontsize=fontsize, offset=0.17, color="b")
    ax.right_axis_label("Green (%)", fontsize=fontsize, offset=0.17, color="g")
    ax.bottom_axis_label("Red (%)", fontsize=fontsize, offset=0.03, color="r")
# -- Plot the entire range.
tax1.ticks(multiple=10, offset=0.02, axes_colors=axes_colors)
tax1.line((25, 25, 0), (65, 25, 0), color='orange', lw=2.0)
tax1.line((25, 25, 0), (25, 65, 0), color='orange', lw=2.0)
tax1.line((25, 65, 0), (65, 25, 0), color='orange', lw=2.0)
tax1.scatter(mlc_per[idx] * 100, c=colors, colormap="viridis", alpha=0.7, s=40,
    label="LightCurve")
# -- Plot the zoomed region.
tax2.set_axis_limits({'b': [25, 65], 'l': [10, 50], 'r': [25, 65]})
tax2.get_ticks_from_axis_limits(multiple=5)
tax2.set_custom_ticks(fontsize=10, offset=0.025, multiple=5,
    axes_colors=axes_colors, tick_formats=tick_formats)
points_c = tax2.convert_coordinates(mlc_per[idx] * 100, axisorder='brl')
tax2.scatter(points_c, c=colors, cmap="viridis", s=40)
# -- Add colorbar
norm = plt.Normalize(vmin=bigoffs.min(), vmax=bigoffs.max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm._A = []
cb = plt.colorbar(sm, ax=ax2, cax=ax3)
cb.set_label("Timesteps To Bigoff")
# -- Reset position of ternary plots and show.
tax1.ax.set_position([0.01, 0.05, 0.41, 0.8])
tax2.ax.set_position([0.44, 0.05, 0.41, 0.8])
tax1.resize_drawing_canvas()
tax2.resize_drawing_canvas()
ternary.plt.show()
