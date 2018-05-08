import tensorflow as tf
import datetime
log = '/home/eeb02/PycharmProjects/kaggle/my_model/my_model.ckpt'


def main():
    file_queue = tf.train.string_input_producer(['/data/kaggle/train.csv'], shuffle=False)

    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)
    value = tf.decode_csv(value, [[1.]] * 66)
    gender, age, TotalGV, Intracranial_volume = value[:4]
    DKT = tf.cast(value[4:], tf.float32)
    label = tf.one_hot(tf.cast(gender - 1, tf.int32), 2)

    x_batch, label_batch = tf.train.shuffle_batch([DKT, label], 121, 484, 0, 1)

    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [x_batch, label_batch], capacity=4)
    dequeue_op = batch_queue.dequeue()

    dropout_rate = tf.placeholder(tf.float32)

    temp = tf.layers.dense(x_batch, 1024, activation=tf.nn.sigmoid)
    temp = tf.layers.dense(temp, 2048, activation=tf.nn.sigmoid)
    temp = tf.layers.dropout(temp, dropout_rate)
    y = tf.layers.dense(temp, 2)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=label_batch, logits=y)
    train = tf.train.AdamOptimizer(0.001).minimize(loss)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(label_batch, 1), predictions=tf.argmax(y, 1))[1]

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir_train = "tensorboard_train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    logdir_test = "tensorboard_test/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer_train = tf.summary.FileWriter(logdir_train, sess.graph)
    writer_test = tf.summary.FileWriter(logdir_test, sess.graph)

    sess.run(init)
    saver.restore(sess, log)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    global_step = 0
    for i in range(10000):
        for j in range(3):
            _, sumery, loss_train = sess.run([train, merged, loss], {dropout_rate: 0.5})
            print('step {}: loss = {}'.format(global_step, loss_train))
            writer_train.add_summary(sumery, global_step)
            global_step += 1
        sumery = sess.run(merged, {dropout_rate: 1})
        writer_test.add_summary(sumery, global_step)
    saver.save(sess, log)
    coord.request_stop()
    coord.join(threads)
    print('complete')


if __name__ == '__main__':
    main()
