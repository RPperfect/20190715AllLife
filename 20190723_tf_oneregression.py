import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

trX=np.linspace(-1,1,101)
trY=2*trX+np.random.rand(*trX.shape)*0.4+0.2
plt.figure()
plt.scatter(trX,trY)
plt.plot(trX,.2+2*trX)

get_ipython().magic(u'matplotlib inline')

X=tf.placeholder("float",name="X")
Y=tf.placeholder("float",name="Y")

with tf.name_scope("Model"):
    
    def model(X,w,b):
        return tf.mul(X,w)
        
    w=tf.Variable(-1.0,name="b0")
    b=tf.Variable(-2.0,name="b1")
    y_model=model(X,w,b)
    
with tf.name_scope("CostFunction"):
    cost=(tf.pow(Y-y_model,2))

train_op=tf.train.GradientdescentOptimizer(0.05).minimize(cost)

sess=tf.Session()
init=tf.initialize_all_variables()
tf.train.write_graph(sess.graph,'/home/ubuntu/linear','graph.pbtxt')
cost_op=tf.scalar_summary("loss",cost)
merged=tf.merge_all_summaries()
sess.run(init)
writer=tf.train.SummaryWriter('/home/ubuntu/linear',sess.graph)

for i in range(100):
    for (x,y) in zip(trX,trY):
        sess.run(train_op,feed_dict={X:x,Y:y})
        summary_str=sess.run(cost_op,feed_dict={X:x,Y:y})
        writer.add_summary(summary_str,i)
        b0temp=b.eval(session=sess)
        b1temp=w.eval(session=sess)
plt.plot(trX,b0temp+b1temp*trX)

print(sess.run(w))
print(sess.run(b))

plt.scatter(trX,trY)
plt.plot(trX,sess.run(b)+trX*sess.run(w))
