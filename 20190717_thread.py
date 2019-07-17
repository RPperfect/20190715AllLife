#Account.py
import threading
import time

class Account:
    def _init_(self,account_no,balance):
        self.account_no=account_no
        self._balance=balance
        self.lock=threading.RLock()
    
    def getBalance(self):
        return self._balance
        
    def draw(self,draw_amount):
        self.lock.acquire()
        try:
            if self._balance>=draw_amount:
                print(threading.current_thread().name+"取钱成功！吐出钞票："+str(draw_amount))
                time.sleep(0.001)
                self._balance-=draw_amount
                print("\t余额为："+str(self._balance))
            else:
                print(threading.current_thread().name+"取钱失败！余额不足！")
        finally:
            self.lock.release()
            
#draw_test.py
import threading
import Account

def draw(account,draw_amount):
    account.draw(draw_amount)

acct=Account.Account("1234567",10000)
threading.Thread(name='甲',target=draw,args=(acct,800)).start()
threading.Thread(name='乙',target=draw,args=(acct,800)).start()
