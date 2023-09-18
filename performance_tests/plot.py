import matplotlib.pyplot as plt
import numpy as np
import mplcursors
with open("results.txt",'r') as f:
    first=f.readline()
    x_axis=first.split(' ')
    second=f.readline()
    y_axis=second.split(' ')
    m=f.readline()
y_axis=[round(float(i),4) for i in y_axis]
fig,ax=plt.subplots()
#plt.xticks(np.arange(len(y_axis)), x_axis)
c = ['tab:red', 'tab:green']
l = ['new','old']
l += ['_'] * (len(y_axis)-len(l))
ax.bar(np.arange(len(y_axis)),y_axis, label=l,color=c)
ax.set_ylabel('seconds')
ax.set_title(f"Time results (5 repeats), multiplier {m}")
ax.legend(title='Legend')

cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
@cursor.connect("add")
def on_add(sel):
    x, y, width, height = sel.artist[sel.index].get_bbox().bounds
    sel.annotation.set(text=f"{x_axis[int(x+0.5)]}: {height}",
                       position=(0, 20), anncoords="offset points")
    sel.annotation.xy = (x + width / 2, y + height)

plt.show()