from loader import *
from model import *
import shutil

batch_size = 100
lr = 1e-1
lr_decay_ratio = 0.95
lr_decay_interval = 500
train_log_interval = 50
valid_log_interval = 1000
train_loader = Loader(data_path=os.path.join("data/train"), default_batch_size=batch_size)
valid_loader = Loader(data_path=os.path.join("data/valid"), default_batch_size=batch_size)

model = Model(batch_size=batch_size)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.allow_soft_placement = True
sess = tf.Session(config=config)

shutil.rmtree('log', ignore_errors=True)
os.makedirs('log')

train_summary_writer = tf.summary.FileWriter(
    'log/train',
    sess.graph,
)

valid_summary_writer = tf.summary.FileWriter(
    'log/valid',
    sess.graph
)
sess.run(tf.global_variables_initializer())

summ = tf.summary.merge_all()

accum_loss = .0
accum_correct_count = .0
accum_conf_matrix = None
train_loader.reset()
while True:
    batch_data = train_loader.get_batch()
    if batch_data is None:
        continue

    sess_input = [
        model.train_op,
        model.cls_loss,
        model.correct_count,
        model.conf_matrix,
        summ,
        model.global_step,
    ]
    sess_output = sess.run(
        fetches=sess_input,
        feed_dict={
            model.lr_placeholder: lr,
            model.labeled_image_ph: batch_data.images,
            model.label_ph: batch_data.labels,
        }
    )

    cur_step = sess_output[-1]
    accum_loss += sess_output[1]
    accum_correct_count += sess_output[2]
    if accum_conf_matrix is None:
        accum_conf_matrix = sess_output[3]
    else:
        accum_conf_matrix += sess_output[3]

    train_summary_writer.add_summary(sess_output[4])
    train_summary_writer.flush()

    if cur_step > 0 and cur_step % train_log_interval == 0:

        loss = accum_loss / train_log_interval
        accuracy = accum_correct_count / (batch_size * train_log_interval)

        print("[step %d] training loss = %f, accuracy = %.6f, lr = %.6f" % (cur_step, loss, accuracy, lr))

        # log for tensorboard
        custom_summaries = [
            tf.Summary.Value(tag='loss', simple_value=loss),
            tf.Summary.Value(tag='accuracy', simple_value=accuracy),
            tf.Summary.Value(tag='learning rate', simple_value=lr),
        ]
        train_summary_writer.add_summary(tf.Summary(value=custom_summaries), cur_step)
        train_summary_writer.flush()

        # reset local accumulations
        accum_loss = .0
        accum_correct_count = .0
        accum_conf_matrix = None

    if cur_step > 0 and cur_step % valid_log_interval == 0:
        valid_loader.reset()

        step_counter = .0
        valid_accum_loss = .0
        valid_accum_correct_count = .0
        valid_accum_conf_matrix = None

        while True:
            batch_data = valid_loader.get_batch()
            if batch_data is None:
                # print('%d validation complete' % self.epoch_counter)
                break

            sess_input = [
                model.cls_loss,
                model.correct_count,
                model.conf_matrix,
            ]
            sess_output = sess.run(
                fetches=sess_input,
                feed_dict={
                    model.labeled_image_ph: batch_data.images,
                    model.label_ph: batch_data.labels,
                }
            )

            valid_accum_loss += sess_output[0]
            valid_accum_correct_count += sess_output[1]
            if valid_accum_conf_matrix is None:
                valid_accum_conf_matrix = sess_output[2]
            else:
                valid_accum_conf_matrix += sess_output[2]

            step_counter += 1

        cur_valid_loss = valid_accum_loss / step_counter
        cur_valid_accuracy = valid_accum_correct_count / (step_counter * batch_size)

        # log for tensorboard
        cur_step = sess.run(model.global_step)
        custom_summaries = [
            tf.Summary.Value(tag='loss', simple_value=cur_valid_loss),
            tf.Summary.Value(tag='accuracy', simple_value=cur_valid_accuracy),
        ]
        valid_summary_writer.add_summary(tf.Summary(value=custom_summaries), cur_step)
        valid_summary_writer.flush()

        print("... validation loss = %f, accuracy = %.6f" % (cur_valid_loss, cur_valid_accuracy))

    if cur_step > 0 and cur_step % lr_decay_interval == 0:
        lr *= lr_decay_ratio
