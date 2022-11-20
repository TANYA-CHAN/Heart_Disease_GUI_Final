from tkinter import *
from PIL import ImageTk,Image
import operator
import KNN_single
import SVM_single
import DT_single

master = Tk ()
master.title ("Detection of Cardiac Arrhythmia")

w, h = master.winfo_screenwidth (), master.winfo_screenheight ()
canvas = Canvas (master, width=w ,height=h, highlightthickness=0)
canvas.place (x=0,y=0)
back = ImageTk.PhotoImage (Image.open('background1.jpg').resize((w,h)))
canvas.create_image (int(w/2),int(h/2), anchor="center", image = back)
canvas.back = back


def page_2 () :

        master2 = Toplevel()
        master2.title ("Detection of Cardiac Arrhythmia")

        w, h = master2.winfo_screenwidth(), master.winfo_screenheight()
        canvas = Canvas(master2, width=w, height=h, highlightthickness=0)
        canvas.place (x=0,y=0)
        back = ImageTk.PhotoImage (Image.open('background2.jpg').resize((w,h)))
        canvas.create_image (int(w/2),int(h/2), anchor="center", image=back)
        canvas.back = back    

        data = []
        def gettext () :
                pred = []
                global data
                data1 = text.get("1.0",'end-1c')
                data = list (data1.split('\t'))
                data = [float(x) for x in data]

                knn = KNN_single.knn (data)
                pred.append (knn)
                knn = 'Class predicted by KNN : ' + (str(knn)).replace('.0','')

                svm = SVM_single.svm (data)
                pred.append (svm)
                svm = 'Class predicted by SVM : ' + (str(svm)).replace('.0','')

                dt = DT_single.Dt (data)
                pred.append (dt)
                dt = 'Class predicted by Decision Tree : ' + (str(dt)).replace('.0','')

                label1.config (text=svm)        
                label2.config (text=dt)
                label3.config (text=knn)

                count = {}
                for i in range (len(pred)) :
                        if (i not in count.keys()) :
                                count.update({pred[i]:pred.count(pred[i])})
                print(count)

                m = max(count.items(), key=operator.itemgetter(1))[0]
                message= 'Most Probable Class : ' + (str(m)).replace('.0','')
                label4.config (text=message)




        label1 = Label (master2, text='', fg="white", bg="#324486", font = ('Segoe UI Semibold',20))
        label1.place(x=1200, y=300)       # for SVM
        label2 = Label (master2, text='', fg="white", bg="#314989", font = ('Segoe UI Semibold',20))
        label2.place(x=1200, y=375)       # for Decision Tree
        label3 = Label (master2, text='', fg="white", bg="#314f8b", font = ('Segoe UI Semibold',20))
        label3.place(x=1200, y=450)       # for KNN
        label4 = Label (master2, text='', fg="white", bg="#31518c", font = ('Segoe UI Semibold',24,'italic','underline'))
        label4.place(x=1200, y=600)       # for result

        enter_image = ImageTk.PhotoImage (Image.open ('button_enter-values-here.png'))
        enter_bg = '#%02x%02x%02x' % (47, 86, 255)
        label = Label (master2, image=enter_image, bg=enter_bg, borderwidth=0, activebackground=enter_bg, highlightthickness=0)
        label.image = enter_image
        label.place (x=510,y=130)

        #entry = Entry (master2, width=200)
        #entry.place (x=50, y = 200)

        text = Text (master2, width = 45, height = 30, bd = 5, wrap = 'word', font = ('Times New Roman',14))
        text.place (x=480, y=220)
        
        done_image = ImageTk.PhotoImage (Image.open('button_done.png'))
        done_bg = '#%02x%02x%02x' % (48,132,168)
        button2 = Button(master2, image=done_image, command = gettext, bg=done_bg, borderwidth=0, cursor='hand2', activebackground=done_bg, highlightthickness=0)
        button2.place (x=575, y=900)

        result_image = ImageTk.PhotoImage (Image.open ('button_result.png'))
        result_bg = '#%02x%02x%02x' % (47, 86, 255)
        label5 = Label (master2, image=result_image, bg=result_bg, borderwidth=0, activebackground=result_bg, highlightthickness=0)
        label5.image = result_image
        label5.place (x=1250,y=130)

        master2.geometry ("%dx%d+0+0" % (w, h))
        mainloop()


upload_button1 = ImageTk.PhotoImage (Image.open('button_get-started.png'))
uploadbg = '#%02x%02x%02x' % (47, 66, 132)
button1 = Button (master, image = upload_button1, command = page_2, bg=uploadbg, borderwidth=0, cursor='hand2', activebackground=uploadbg, highlightthickness=0)
button1.place (x=1600, y=900)


master.geometry ("%dx%d+0+0" % (w, h))
mainloop()