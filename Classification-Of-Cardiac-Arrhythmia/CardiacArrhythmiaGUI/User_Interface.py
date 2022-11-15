from tkinter import *
from PIL import ImageTk, Image
import Naive_CV as nb
import KNN_CV as knn
import SVM_CV as svm
import Decision_Trees_CV as dt
import VFI_Median_cross_valid as vfi1
import VFI2_Median_cross_valid as vfi2
import VFI3_Median_cross_valid as vfi3
import VFI4_Median_cross_valid as vfi4
import VFI5_Median_cross_valid as vfi5
import LDA_crossvalid as lda
def ui():

    master = Tk()
    # button1 = Button(master, text="Naive Bayes", command=nb.naive_bayes, height=2, width=10)
    # button1.place(x=200, y=50)

    button2 = Button(master, text="KNN", command=knn.knn, height=2, width=10)
    button2.place(x=200, y=100)

    button3 = Button(master, text="SVM", command=svm.svm, height=2, width=10)
    button3.place(x=200, y=150)

    button3 = Button(master, text="Decision Tree", command=dt.decision_tree, height=2, width=10)
    button3.place(x=200, y=200)

    # button4 = Button(master, text="VFI", command=vfi1.vfi, height=2, width=10)
    # button4.place(x=200, y=250)

    # button5 = Button(master, text="VFI2", command=vfi2.vfi, height=2, width=10)
    # button5.place(x=200, y=300)

    # button6 = Button(master, text="VFI3", command=vfi3.vfi, height=2, width=10)
    # button6.place(x=200, y=350)

    # button7 = Button(master, text="VFI4", command=vfi4.vfi, height=2, width=10)
    # button7.place(x=200, y=400)

    # button8 = Button(master, text="VFI5", command=vfi5.vfi, height=2, width=10)
    # button8.place(x=200, y=450)

    # button8 = Button(master, text="LDA", command=lda.lda_main, height=2, width=10)
    # button8.place(x=200, y=500)
    w, h = master.winfo_screenwidth(), master.winfo_screenheight()
    master.geometry("%dx%d+0+0" % (w, h))

    master.geometry("800x800")

    #mainloop()


