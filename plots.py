
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np 
from datetime import datetime, timedelta
from matplotlib import rcParams, rcParamsDefault

def generate_bars(vals, output_path):
    font_properties = FontProperties(fname="Roboto-Regular.ttf")
    bold_font_properties = FontProperties(fname="Roboto-Bold.ttf")
    plt.rcParams["figure.figsize"] = [7.50, 1.75]
    plt.rcParams["figure.autolayout"] = True

    data = ["Data"]
    # add data here of all the average glucose values in the week and make percentages according to the categories
    less_50 = [vals[0]]
    less_80 = [vals[1]]
    less_150 = [vals[2]]
    less_190 = [vals[3]]

    # Create the bar plots
    b1 = plt.barh(data, less_50, color="#F3E4FF")
    b2 = plt.barh(data, less_80, left=less_50[0], color="#B5C5FF")
    b3 = plt.barh(data, less_150, left=less_50[0] + less_80[0], color="#CAFFE2")
    b4 = plt.barh(data, less_190, left=less_50[0] + less_80[0] + less_150[0], color="#FFDEC5")

    # Set the legend
    plt.axis(False)

    labels = ["< 50", "< 80", "< 150", "< 190"]
    values = [less_50[0], less_80[0], less_150[0], less_190[0]]

    for i, bar in enumerate(b4):
        value = values[i]
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f"{value}%", ha="center", va="center", color="black", fontproperties=bold_font_properties, weight="bold")
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2- 0.1 , labels[0], ha="center", va="center", color="black",
                fontproperties=font_properties, weight="bold")

    for i, bar in enumerate(b3):
        value = values[i]
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f"{value}%", ha="center", va="center", color="black", fontproperties=bold_font_properties, weight="bold")
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2 - 0.1 , labels[1], ha="center", va="center", color="black",
                fontproperties=font_properties, weight="bold")

    for i, bar in enumerate(b2):
        value = values[i]
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f"{value}%", ha="center", va="center", color="black", fontproperties=bold_font_properties, weight="bold")
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2 - 0.1, labels[2], ha="center", va="center", color="black",
                fontproperties=font_properties, weight="bold")

    for i, bar in enumerate(b1):
        value = values[i]
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f"{value}%", ha="center", va="center", color="black", fontproperties=bold_font_properties, weight="bold")
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2 - 0.1, labels[3], ha="center", va="center", color="black",
                fontproperties=font_properties, weight="bold")

    plt.savefig(output_path,bbox_inches='tight', pad_inches = 0) # save file
    rcParams.update(rcParamsDefault)

def generate_bpm(vals, output_path):
    font_properties=FontProperties(fname='Roboto-Regular.ttf')

    def last_7_days_dates():
        today = datetime.today()
        return (today - timedelta(days=6)).strftime('%d')

    def plot_values(arr):
        plt.figure()
        plt.xlim(-0.5,7)
        plt.ylim(0, 120)  
        plt.xticks(np.arange(7),[str(int(initial_day)+i) for i in range(0,7)])

        plt.title("RESTING HEART RATE",fontproperties=font_properties,fontsize=20)
        plt.plot(arr, '--k')
        plt.plot(arr,"o",color="#FF5252")
        

    initial_day=int(last_7_days_dates())

    plot_values(vals)

    plt.savefig(output_path)
    rcParams.update(rcParamsDefault)

def generate_calories(vals, output_path):
    font_properties=FontProperties(fname='Roboto-Regular.ttf')

    def last_7_days_dates():
        today = datetime.today()
        return (today - timedelta(days=6)).strftime('%d')

    def plot_values(arr):
        plt.figure()
        plt.xlim(-0.5,7)
        plt.ylim(0, 3500)  
        plt.xticks(np.arange(7),[str(int(initial_day)+i) for i in range(0,7)])

        plt.title("CALORIES INTAKE",fontproperties=font_properties,fontsize=20)
        plt.plot(arr, '--k')
        plt.plot(arr,"o",color="#FF5252")
    

    initial_day=int(last_7_days_dates())

    plot_values(vals)

    plt.savefig(output_path)
    rcParams.update(rcParamsDefault)

def generate_glucose(vals, output_path):
    font_properties=FontProperties(fname='Roboto-Regular.ttf')


    def get_color_gradient(c1, c2, n):
        assert n > 1
        c1_rgb = np.array(hex_to_RGB(c1))/255
        c2_rgb = np.array(hex_to_RGB(c2))/255
        mix_pcts = [x/(n-1) for x in range(n)]
        rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
        return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


    def hex_to_RGB(hex_str):
        """ #FFFFFF -> [255,255,255]"""
        #Pass 16 to the integer function for change of base
        return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

    def loop_gradient(c1,c2,initial_val):
        grad_arr=get_color_gradient(c1,c2,10)           #generate a list of color values for gradient
        for i in range(0,10):
            plt.axhspan(i+initial_val, initial_val+i+1, facecolor=grad_arr[i], alpha=1)

    def last_7_days_dates():
        today = datetime.today()
        return (today - timedelta(days=6)).strftime('%d')

    def plot_values(arr):
        plt.figure()
        plt.xlim(-0.5,7)
        plt.ylim(0, 350)  
        plt.xticks(np.arange(7),[str(int(initial_day)+i) for i in range(0,7)])


        plt.axhspan(0, 45, facecolor="#F2E1FF", alpha=1)
        loop_gradient("#F2E1FF","#B5C5FF",45)
        plt.axhspan(55, 75, facecolor="#B5C5FF", alpha=1)
        loop_gradient("#B5C5FF","#CAFFE2",75)
        plt.axhspan(85, 145, facecolor="#CAFFE2", alpha=1)
        loop_gradient("#CAFFE2","#FFDDC5",145)
        plt.axhspan(155, 185, facecolor="#FFDDC5", alpha=1)
        loop_gradient("#FFDDC5","#FCCECE",185)
        plt.axhspan(195, 350, facecolor="#FCCECE", alpha=1)

        plt.title("WEEKLY GLUCOSE OVERLAY",fontproperties=font_properties,fontsize=20)
        plt.plot(arr, '--k')
        plt.plot(arr,"o",color="#FF5252")
    

    initial_day=int(last_7_days_dates())

    plot_values(vals)

    plt.savefig(output_path)
    rcParams.update(rcParamsDefault)

