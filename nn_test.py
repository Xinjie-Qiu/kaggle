import tensorflow as tf
import csv
import numpy as np
log = '/home/eeb02/PycharmProjects/kaggle/my_model/my_model.ckpt'


def main():
    file_queue = tf.train.string_input_producer(['/data/kaggle/test.csv'], shuffle=False)

    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)
    value = tf.decode_csv(value, [[1.]] * 66)
    id, age, TotalGV, Intracranial_volume = value[:4]
    DKT = tf.cast(value[4:], tf.float32)
    label = tf.cast(id, tf.int32)

    x_batch, label_batch = tf.train.batch([DKT, label], 83, 1, 83)

    temp = tf.layers.dense(x_batch, 1024, activation=tf.nn.sigmoid)
    temp = tf.layers.dense(temp, 2048, activation=tf.nn.sigmoid)
    y = tf.layers.dense(temp, 2)
    test = (label_batch, tf.argmax(y, 1) + 1)

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    saver.restore(sess, log)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    test_3 = sess.run(test)
    test_3 = np.array(test_3).transpose((1, 0))

    title = ['id', 'gender']

    f = open("stock.csv", "w")
    w = csv.writer(f)

    w.writerow(title)

    for line in test_3:
        w.writerow(line)
    f.close()

    coord.request_stop()
    coord.join(threads)
    print('yah')


if __name__ == '__main__':
    main()