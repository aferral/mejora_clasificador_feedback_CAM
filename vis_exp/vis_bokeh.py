import numpy as np

from matplotlib import cm
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh import plotting as bplot
from bokeh.models import (LassoSelectTool, PanTool,
                          ResizeTool, ResetTool,
                          HoverTool, WheelZoomTool,BoxZoomTool)
TOOLS = [LassoSelectTool, PanTool, WheelZoomTool, ResizeTool, ResetTool, BoxZoomTool]
import pandas as pd
import os
import random

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


def plotBlokeh(points,labels,indexs,name_out,outputFolder, test=False):

    host='http://192.168.100.6:8000'


    assert (len(points.shape) == 2)
    assert(points.shape[1] == 2)

    if test:
        points = np.random.rand(1000, 2)
        labels = np.random.randint(0, 3, size=(1000))
        indexs = ['test' for i in range(1000)]

    tooltip = """
        <div>
            <div>
                <img
                src="@image_urls" width=200 height=200 alt="image"
                style="float: left; margin: 0px 15px 15px 0px; image-rendering: pixelated;"
                border="2"
                ></img>
            </div>
            <div>
                <span style="font-size: 17px;">index: @img_index label: @label</span>
            </div>
        </div>
              """

    #POINTS,LABELS,IMAGES
    nPoints = points.shape[0]

    # if nPoints > 1000:
    #     points = points[0:1000]
    #     labels = labels[0:1000]
    #     indexs = indexs[0:1000]
    #     nPoints = 1000

    df = pd.DataFrame(index=np.arange(nPoints), columns={'z','w','image_urls','img_index','label'})

    df['z'] = pd.Series(points[:,0], index=df.index)
    df['w'] = pd.Series(points[:,1], index=df.index)


    df['image_urls'] = ["{0}/{1}".format(host,indexs[i]) for i in range(len(indexs))]
    df['img_index'] = indexs
    df['label'] = labels

    bplot.output_file(os.path.join(outputFolder,name_out+'.html'))
    hover0 = HoverTool(tooltips=tooltip)

    tools0 = [t() for t in TOOLS] + [hover0]

    p = figure(plot_width=800, plot_height=800, tools= tools0,
               title="Mouse over the dots")

    colorList = []
    nClases = len(set(df['label']))
    print("There are ",nClases," classes")

    for i in range(nClases):
        nc = generate_new_color(colorList, pastel_factor=0.9)
        colorList.append(nc)
    for i in range(len(colorList)):
        colorList[i] = [int(colorList[i][0] * 255), int(colorList[i][1] * 255), int(colorList[i][2] * 255)]
    colorList =  ["#{0:02x}{1:02x}{2:02x}".format(x[0],x[1],x[2]) for x in colorList]



    colorList = np.array(colorList)


    colors = colorList[df['label'].values]
    p.scatter(source=df, x='z', y='w', fill_color=colors,size=10 )
    show(p)

if __name__ == '__main__':
    plotBlokeh(np.random.rand(100,2), np.random.randint(0, 3, size=(100)), ['a' for i in range(100)], 'test', './', test=True)
