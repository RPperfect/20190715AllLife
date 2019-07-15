# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:16:43 2019

@author: ruipeng
"""
#first_thread.py
import threading
def action(max):
    for i in range(max):
        print(threading.current_thread().getName() + " " + str(i))

for i in range(100):
    print(threading.current_thread().getName() + " " + str(i))
    if i==20:
        t1=threading.Thread(target=action,args=(100,))
        t1.start()
        t2=threading.Thread(target=action,args=(100,))
        t2.start()
print('主线程执行完成！')

#second_thread.py
import threading
class FkThread(threading.Thread):
    def _init_(self):
        threading.Thread.__init__(self)
        self.i=0
        
    def run(self):
        while self.i<100:
            print(threading.current_thread().getName()+" "+str(self.i))
            self.i+=1
            
for i in range(100):
    print(threading.current_thread().getName()+" "+str(i))
    if i==20:
        ft1=FkThread()
        ft1.start()
        ft2=FkThread()
        ft2.start()
print('主线程执行完成！')

#invoke_run.py
import threading
def action(max):
    for i in range(max):
        print(threading.current_thread().name+" "+str(i))
        
for i in range(100):
    print(threading.current_thread().name+" "+str(i))
    if i==20:
        threading.Thread(target=action,args=(100,)).run()
        threading.Thread(target=action,args=(100,)).run()
        
#start_dead.py
import threading
def action(max):
    for i in range(100):
        print(threading.current_thread().name+" "+str(i))
        
sd=threading.Thread(target=action,args=(100,))
for i in range(300):
    print(threading.current_thread().name+" "+str(i))
    if i==20:
        sd.start()
        print(sd.is_alive())
    if i>20 and not(sd.is_alive()):
        sd.start()
        
#join_thread.py
import threading
def action(max):
    for i in range(max):
        print(threading.current_thread().name+" "+str(i))
        
threading.Thread(target=action,args=(100,),name="新线程").start()
for i in range(100):
    if i==20:
        jt=threading.Thread(target=action,args=(100,),name="被Join的线程")
        jt.start()
        jt.join()
    print(threading.current_thread().name+" "+str(i))
    
#daemon_thread.py
import threading
def action(max):
    for i in range(max):
        print(threading.current_thread().name+" "+str(i))

t=threading.Thread(target=action,args=(100,),name='后台线程')
t.daemon=true
t.start()
for i in range(10):
    print(threading.current_thread().name+" "+str(i))
    
#sleep_test.py
import time
for i in range(10):
    print("当前时间:%s"%time.ctime())
    time.sleep(1)