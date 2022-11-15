from tkinter import *
import operator
def page2():
    content =[]
    import VFI_single
    import KNN_single
    import SVM_single
    import Naive_single
    import VFI5_single
    import VFI4_single
    import VFI3_single
    import VFI2_single
    import DT_single
    import LDA_single
    def gettext():
        pred =[]
        print('Correct till here 3')
        global content
        content1 = entry.get()
        content = list(content1.split('\t'))
        content = [float(x) for x in content]
        vfi1 = VFI_single.vfi(content)
        pred.append(vfi1)
        # vfi1 = 'The class predicted by VFI1 is: '+str(vfi1)
        # print(vfi1)
        knn = KNN_single.knn(content)
        print("KNN_single has returned = ", knn)
        pred.append(knn)
        knn = 'The type of Arrythmia class predicted by KNN is: '+str(knn)
        svm = SVM_single.svm(content)
        pred.append(svm)
        svm = 'The type of Arrythmia class predicted by SVM is: '+str(svm)
        # nb = Naive_single.nb(content)
        # pred.append(nb)
        # nb = 'The class predicted by Naive Bayes is: '+str(nb)
        # Vfi5 = VFI5_single.vfi5(content)
        # pred.append(Vfi5)
        # Vfi5 = 'The class predicted by VFI5 is: '+str(Vfi5)
        dt = DT_single.Dt(content)
        pred.append(dt)
        dt = 'The type of Arrythmia class predicted by Decision Tree is: '+str(dt)
        # Vfi2 = VFI2_single.vfi2(content)
        # pred.append(Vfi2)
        # Vfi2 = 'The class predicted by VFI2 is: '+str(Vfi2)
        # Vfi3 = VFI3_single.vfi3(content)
        # pred.append(Vfi3)
        # Vfi3 = 'The class predicted by VFI3 is: '+str(Vfi3)
        # Vfi4 = VFI4_single.vfi4(content)
        # pred.append(Vfi4)
        # Vfi4 = 'The class predicted by VFI4 is: '+str(Vfi4)
        # Lda = LDA_single.lda(content)
        # pred.append(Lda)
        # Lda = 'The class predicted by LDA is: '+str(Lda)
        # label3.config(text=vfi1)
        label6.config(text=knn)
        label4.config(text=svm)
        # label2.config(text=nb)
        # label10.config(text=Vfi5)
        label5.config(text=dt)
        # label7.config(text=Vfi2)
        # label8.config(text=Vfi3)
        # label9.config(text=Vfi4)
        # label11.config(text=Lda)
        count={}
        for i in range(len(pred)):
            if(i not in count.keys()):
                count.update({pred[i]:pred.count(pred[i])})
        print(count)
        m = max(count.items(), key=operator.itemgetter(1))[0]
        message= 'MOST PROBABLE CLASS : ' + str(m)
        label12.config(text=message)
            

    master = Tk()
    # label1 = Label(master, text="Enter the data:", height=7, width=50)
    # label1.place(x=500, y=100)
    # label2 = Label(master, text='', height=2, width=50)
    # label2.place(x=500, y=300)
    # label3 = Label(master, text='', height=2, width=50)
    # label3.place(x=500, y=400)
    label4 = Label(master, text='', height=2, width=50)
    label4.place(x=500, y=325)
    label5 = Label(master, text='', height=2, width=50)
    label5.place(x=500, y=350)
    label6 = Label(master, text='', height=2, width=50)
    label6.place(x=500, y=375)
    # label7 = Label(master, text='', height=2, width=50)
    # label7.place(x=500, y=425)
    # label8 = Label(master, text='', height=2, width=50)
    # label8.place(x=500, y=450)
    # label9 = Label(master, text='', height=2, width=50)
    # label9.place(x=500, y=475)
    # label10 = Label(master, text='', height=2, width=50)
    # label10.place(x=500, y=500)
    # label11 = Label(master, text='', height=2, width=50)
    # label11.place(x=500, y=525)
    label12 = Label(master, text='', height=2, width=50)
    label12.place(x=500, y=600)
    entry = Entry(master, width=200)
    entry.place(x=50, y = 200)
    button1 = Button(master, text="Done",command = gettext,height=2, width=50)
    button1.place(x=500, y=250)
    w, h = master.winfo_screenwidth(), master.winfo_screenheight()
    master.geometry("%dx%d+0+0" % (w, h))

    mainloop()
