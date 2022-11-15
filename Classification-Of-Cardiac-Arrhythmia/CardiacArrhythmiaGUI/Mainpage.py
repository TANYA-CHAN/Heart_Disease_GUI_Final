from tkinter import *

master = Tk()
import User_Interface as ofnew
import Page2 as p2
def askOffsets():
    print('Correct till here')
    of = ofnew.ui()
def callpage2():
    print('Correct till here2')
    of = p2.page2()
button1 = Button(master, text="Please click here to check the type of Arrhythmia for single patient",  command = callpage2, height=5, width=80)
button1.place(x=500, y=300)

# button2 = Button(master, text="Check accuracies of algorithms",  command = askOffsets, height=2, width=50)
# button2.place(x=500, y=300)

label1 = Label(master, text="CLASSIFICATION OF CARDIAC ARRHYTHMIA!", height=7, width=50)
label1.place(x=500, y=100)
w, h = master.winfo_screenwidth(), master.winfo_screenheight()
master.geometry("%dx%d+0+0" % (w, h))
mainloop()
