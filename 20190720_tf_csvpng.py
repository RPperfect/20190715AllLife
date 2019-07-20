#CSV.py
import tensorflow as tf
sess=tf.Session()
filename_queue=tf.train.string_input_producer(tf.train.match_filenames_once("./*".csv),shuffle=True)
reader=tf.TextLineReader(skip_header_line=1)
key,value=reader.read(filename_queue)
record_defaults=[[0.],[0.],[0.],[0.],[""]]
col1,col2,col3,col4,col5=tf.decode_csv(value,record_defaults=record_defaults)
features=tf.pack([col1,col2,col3,col4])

tf.initialize_all_variables().run(session=sess)
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord,sess=sess)

for iteration in range(0,5):
    example=sess.run([features])
print(example)
coord.request_stop()
coord.join(threads)

#PNG.p                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      y
import tensorflow as tf
sess=tf.Session()
filename_queue=tf.train.string_input_producer(tf.train.match_filenames_once("./blue_jay.jpg"))
reader=tf.WholeFileReader()
key,value=reader.read(filename_queue)
image=tf.image.decode_jpeg(value)
flipImageUpDown=tf.image.encode_jpeg(tf.image.flip_up_down(image))
flipImageLeftRight=tf.image.encode_jpeg(tf.image.flip_left_right(image))
tf.initialize_all_variables().run(session=sess)
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(coord=coord,sess=sess)
example=sess.run(flipImageLeftRight)
print(example)
file=open("flippedUpDown.jpg","wb+")
file.write(flipImageUpDown.eval(session=sess))
file.close()
file=open("flippedLeftRight.jpg","wb+")
file.write(flipImageLeftRight.eval(session=sess))
file.close()
