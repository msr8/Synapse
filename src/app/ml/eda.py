# Agg is a non-GUI backend. Helps mitigate the gi required error, which happens in absence of pygobject which is needed by the default matplotlib backend in linux
# It is called so early cause when seaborn's code is executed during import, it may call the default backend at that time and use it in pairplot
import matplotlib
matplotlib.use('Agg')
from rich import inspect

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix

from pygal import Pie, Bar
from pygal.style import NeonStyle
import pandas as pd

from matplotlib import rcParams, style, use
from matplotlib.figure import Figure
from seaborn import kdeplot, heatmap, pairplot, histplot, color_palette

import pandas as pd
import numpy as np

from base64 import b64encode
from io import BytesIO

style.use('dark_background')
# print(style.available)
# print(rcParams['figure.dpi'])
rcParams['figure.dpi'] = 200
# rcParams['figure.dpi'] = 700
# So that we dont have to call `plt.tight_layout()` every time
rcParams['figure.autolayout'] = True
# plt.rcParams['savefig.bbox'] = 'tight'
# Set size of points in scatter plot
rcParams['lines.markersize'] = 2
# Set size of lines in regplots (which are inside the pairplot)
rcParams['lines.linewidth'] = 1


col_palette = color_palette('Set2')
col_palette_alpha = [ (r,g,b,0.6) for r,g,b in col_palette ]

NeonStyle.transition        = '0.3s ease-out'
NeonStyle.background        = '#000000'
NeonStyle.plot_background   = '#000000'
NeonStyle.foreground        = 'rgba(200, 200, 200, 1)'
NeonStyle.foreground_strong = 'rgba(255, 255, 255, 1)'
NeonStyle.foreground_subtle = 'rgb(200,200,200)' # Number of instances when you hover over a pie
NeonStyle.opacity           = 0.6
NeonStyle.opacity_hover     = 1
NeonStyle.colors            = [ f'rgb({int(r*256)},{int(g*256)},{int(b*256)})' for r,g,b in col_palette ]
# inspect(NeonStyle)

BarNeonStyle = NeonStyle()
BarNeonStyle.opacity = 1
BarNeonStyle.opacity_hover = 1
BarNeonStyle.colors = ['#b2f7ef']
BarNeonStyle.stroke_width = 0
BarNeonStyle.stroke_width_hover = 0

'''
cmap -> coolwarm | Blues
palette -> pastel | dark | Set2 | bright | hls
'''





# Correlation matrix heatmap
def corr_heatmap(df:pd.DataFrame) -> str:
    ax = Figure().add_subplot(1,1,1)
    heatmap(df.corr(), cmap='coolwarm', ax=ax, annot=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={'size': 8})
    ax.set_title('Correlation Matrix')
    # Set font size of x and y tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)
    
    img = BytesIO()
    # fig.tight_layout()
    ax.figure.savefig(img, format='png')
    
    return b64encode(img.getvalue()).decode('utf8')


# MI heatmap
def mi_heatmap(df:pd.DataFrame, target:str) -> str:
    x = df.drop(target, axis=1)
    y = df[target]
    
    mi = mutual_info_classif(x, y.astype('str')) # Returns an array ; `astype('str')` cause MI doesn't work with continuous data
    mi = dict(zip(x.columns,mi)) # column -> MI value

    ax = Figure().add_subplot(1,1,1)
    heatmap(pd.DataFrame(mi, index=[0]), cmap='coolwarm', ax=ax, annot=True, fmt='.2f', vmax=1, vmin=0)
    ax.set_yticklabels([''])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f'Mutual Information with target column "{target}"')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Mutual Information')
    
    img = BytesIO()
    # fig.tight_layout()
    ax.figure.savefig(img, format='png')

    return b64encode(img.getvalue()).decode('utf8')


# Pairplot
def pairplot_chart(df:pd.DataFrame, target:str) -> str:
    # ax = Figure().add_subplot(1,1,1)
    fig = pairplot(df, hue=target, kind='reg', diag_kind='kde', palette='bright').figure
    
    img = BytesIO()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    # ax.figure.savefig(img, format='png')
    fig.savefig(img, format='png')

    return b64encode(img.getvalue()).decode('utf8')





# Confusion matrix (during bayesian optimization)
def confusion_matrix_chart(clf, x_test, y_test, classes:list[str], model_dn:str, normalize:str=None) -> str:
    y_pred = clf.predict(x_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    ax = Figure().add_subplot(1,1,1)
    fmt = 'd' if normalize is None else '.2f'
    heatmap(cm, annot=True, fmt=fmt, cmap='Blues', cbar=False, ax=ax, xticklabels=classes, yticklabels=classes, annot_kws={'size': 30})

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(model_dn)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    img = BytesIO()
    # fig.tight_layout()
    ax.figure.savefig(img, format='png')
    return b64encode(img.getvalue()).decode('utf8')






def handle_column(df:pd.DataFrame, col_name:str, target:str, n_unique_threshold:int=10) -> Bar | Pie:
    '''
    If numerical, bin chart (histogram) of N_BINS bins
    If categorical, pie chart
    '''

    data = df[col_name]
    n_unique  = data.nunique()
    n_classes = df[target].nunique()

    # If numerical values (bin chart)
    if pd.api.types.is_numeric_dtype(data) and n_unique >= n_unique_threshold:
        ax = Figure().add_subplot(1,1,1)

        # For the target column, we will use histplot ig
        if col_name != target: kdeplot(df, x=col_name, hue=target, fill=True, common_norm=False, alpha=0.5, palette=col_palette, ax=ax)
        else: histplot(df, x=col_name, fill=True, common_norm=False, alpha=0.5)
        img = BytesIO()
        ax.figure.savefig(img, format='png')

        return {'type': 'matplotlib', 'data': b64encode(img.getvalue()).decode('utf8')}

        # chart       = Bar(style=BarNeonStyle, show_legend=False, title=col_name, stroke=False)
        # binned_data = pd.cut(data, n_bins).value_counts(sort=False)
        # ranges      = binned_data.index.astype(str).to_list()
        # values      = binned_data.values.tolist()
        # for range_,values in zip(ranges, values):
        #     chart.add(range_, values)


    # If categorical values
    else:
        chart = Pie(style=NeonStyle, title=col_name)
        # `counts` is a series with the value as index and counts (no. of instances of that value) as values
        counts = data.value_counts()
        for idx in counts.index: chart.add(str(idx), counts[idx])
    
        return {'type': 'pygal', 'data': chart.render_data_uri()}

        # # Use a seaborn/matplotlib pie chart instead
        # ax = Figure().add_subplot(1,1,1)
        # ax.pie(data.value_counts(), labels=data.value_counts().index, autopct='%1.1f%%', startangle=90, colors=col_palette_alpha, textprops={'color': 'white'}, wedgeprops={'edgecolor': 'black'})
        # ax.legend(title=col_name, loc='upper right', fontsize=8, title_fontsize=10)
        # ax.axis('equal')
        # ax.set_title(col_name)
        # img = BytesIO()
        # ax.figure.savefig(img, format='png')

        # return {'type': 'matplotlib', 'data': b64encode(img.getvalue()).decode('utf8')}
    
        








def eda(data:pd.DataFrame, n_unique_threshold:int=10, n_bins=30) -> list[Bar | Pie]:
    # ret = []
    # for col_name in data.columns:
    #     chart = handle_column(data[col_name], col_name, n_unique_threshold, n_bins)
    #     ret.append(chart)
    # return ret

    return list(map(
        lambda col_name: handle_column(data[col_name], col_name, n_unique_threshold, n_bins),
        data.columns
    ))








