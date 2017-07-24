from uniform_loader import *
from mnist_loader import *
from model import *
import shutil

tf.app.flags.DEFINE_string('data_path', 'data', 'Directory path to read the data files')
tf.app.flags.DEFINE_string('checkpoint_path', 'model', 'Directory path to save checkpoint files')

tf.app.flags.DEFINE_integer('batch_size', 100, 'mini-batch size for training')

tf.app.flags.DEFINE_float('lr', 1e-4, 'initial learning rate')
tf.app.flags.DEFINE_float('lr_decay_ratio', 0.95, 'ratio for decaying learning rate')
tf.app.flags.DEFINE_integer('lr_decay_interval', 500, 'step interval for decaying learning rate')
tf.app.flags.DEFINE_integer('train_log_interval', 50, 'step interval for triggering print logs of train')
tf.app.flags.DEFINE_integer('valid_log_interval', 500, 'step interval for triggering validation')

ImageInfo = collections.namedtuple("ImageInfo", ['width', 'height', 'channel'])
mnist_image_info = ImageInfo(width=28, height=28, channel=1)
cifar10_image_info = ImageInfo(width=32, height=32, channel=3)

FLAGS = tf.app.flags.FLAGS

print("Learning rate = %e" % FLAGS.lr)


class Learner:
    def __init__(self):
        self.img_info = cifar10_image_info
        self.model = Model(batch_size=FLAGS.batch_size, image_info=self.img_info)

        self.data_path = FLAGS.data_path
        self.batch_size = FLAGS.batch_size

        self.lr = FLAGS.lr
        self.lr_decay_interval = FLAGS.lr_decay_interval
        self.lr_decay_ratio = FLAGS.lr_decay_ratio
        self.train_log_interval = FLAGS.train_log_interval
        self.valid_log_interval = FLAGS.valid_log_interval

        # NOTE : Data = CIFAR-10
        # shutil.rmtree(FLAGS.data_path, ignore_errors=True)
        # os.mkdir(FLAGS.data_path)
        # split_dataset(os.path.join(self.data_path, "train"), "data/sampled_train", ratio=0.02)
        # self.train_labeled_loader = UniformLoader(
        #     data_path="data/sampled_train",
        #     image_info=self.img_info,
        #     default_batch_size=self.batch_size)
        # self.train_unlabeled_loader = UniformLoader(
        #     data_path=os.path.join(self.data_path, "train"),
        #     image_info=self.img_info,
        #     default_batch_size=self.batch_size)
        # self.valid_loader = Loader(
        #     data_path=os.path.join(self.data_path, "val"),
        #     image_info=self.img_info,
        #     default_batch_size=self.batch_size)

        # NOTE : Data = MNIST
        self.train_labeled_loader = MnistLoader(
            data_path=self.data_path,
            default_batch_size=self.batch_size,
            image_info=self.img_info,
            dataset="test")
        self.train_unlabeled_loader = MnistLoader(
            data_path=self.data_path,
            default_batch_size=self.batch_size,
            image_info=self.img_info,
            dataset="train")
        self.valid_loader = MnistLoader(
            data_path=self.data_path,
            default_batch_size=self.batch_size,
            image_info=self.img_info,
            dataset="validation")

        self.sess = tf.Session()

        shutil.rmtree('log', ignore_errors=True)
        os.makedirs('log')

        self.train_summary_writer = tf.summary.FileWriter(
            'log/train',
            self.sess.graph,
        )

        self.valid_summary_writer = tf.summary.FileWriter(
            'log/valid',
            self.sess.graph
        )
        self.sess.run(tf.global_variables_initializer())

        self.summ = tf.summary.merge_all()

    def train(self):
        accum_loss = .0
        accum_correct_count = .0
        accum_conf_matrix = None
        self.train_labeled_loader.reset()
        self.train_unlabeled_loader.reset()
        while True:
            batch_labeled_data = self.train_labeled_loader.get_batch()
            batch_unlabeled_data = self.train_unlabeled_loader.get_batch()
            if (batch_labeled_data is None) or (batch_unlabeled_data is None):
                continue

            sess_input = [
                self.model.train_op,
                self.model.loss,
                self.model.walker_loss,
                self.model.visit_loss,
                self.model.cls_loss,
                self.model.correct_count,
                self.model.conf_matrix,
                self.summ,
                self.model.global_step,
            ]
            sess_output = self.sess.run(
                fetches=sess_input,
                feed_dict={
                    self.model.lr_placeholder: self.lr,
                    self.model.labeled_image_ph: batch_labeled_data.images,
                    self.model.label_ph: batch_labeled_data.labels,
                    self.model.unlabeled_image_ph: batch_unlabeled_data.images,
                }
            )

            cur_step = sess_output[-1]
            accum_loss += sess_output[1]
            accum_correct_count += sess_output[-4]
            if accum_conf_matrix is None:
                accum_conf_matrix = sess_output[-3]
            else:
                accum_conf_matrix += sess_output[-3]

            self.train_summary_writer.add_summary(sess_output[-2], cur_step)
            self.train_summary_writer.flush()

            if cur_step > 0 and cur_step % self.train_log_interval == 0:

                loss = accum_loss / self.train_log_interval
                accuracy = accum_correct_count / (self.batch_size * self.train_log_interval)

                print("[step %d] training loss = %f, accuracy = %.6f, lr = %.6f" % (cur_step, loss, accuracy, self.lr))
                # log for tensorboard
                custom_summaries = [
                    tf.Summary.Value(tag='loss', simple_value=loss),
                    tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                    tf.Summary.Value(tag='learning rate', simple_value=self.lr),
                ]
                self.train_summary_writer.add_summary(tf.Summary(value=custom_summaries), cur_step)
                self.train_summary_writer.flush()

                # reset local accumulations
                accum_loss = .0
                accum_correct_count = .0
                accum_conf_matrix = None

            if cur_step > 0 and cur_step % self.valid_log_interval == 0:
                self.valid_loader.reset()

                step_counter = .0
                valid_accum_cls_loss = .0
                valid_accum_correct_count = .0
                valid_accum_conf_matrix = None

                while True:
                    batch_labeled_data = self.valid_loader.get_batch()
                    if batch_labeled_data is None:
                        # print('%d validation complete' % self.epoch_counter)
                        break

                    sess_input = [
                        self.model.cls_loss,
                        self.model.correct_count,
                        self.model.conf_matrix,
                    ]
                    sess_output = self.sess.run(
                        fetches=sess_input,
                        feed_dict={
                            self.model.labeled_image_ph: batch_labeled_data.images,
                            self.model.label_ph: batch_labeled_data.labels,
                        }
                    )

                    valid_accum_cls_loss += sess_output[0]
                    valid_accum_correct_count += sess_output[-2]
                    if valid_accum_conf_matrix is None:
                        valid_accum_conf_matrix = sess_output[-1]
                    else:
                        valid_accum_conf_matrix += sess_output[-1]

                    step_counter += 1

                cur_valid_loss = valid_accum_cls_loss / step_counter
                cur_valid_accuracy = valid_accum_correct_count / (step_counter * self.batch_size)

                # log for tensorboard
                cur_step = self.sess.run(self.model.global_step)
                custom_summaries = [
                    tf.Summary.Value(tag='loss/classification', simple_value=cur_valid_loss),
                    tf.Summary.Value(tag='accuracy', simple_value=cur_valid_accuracy),
                ]
                self.valid_summary_writer.add_summary(tf.Summary(value=custom_summaries), cur_step)
                self.valid_summary_writer.flush()

                print("... validation loss = %f, accuracy = %.6f" % (cur_valid_loss, cur_valid_accuracy))

            if cur_step > 0 and cur_step % self.lr_decay_interval == 0:
                self.lr *= self.lr_decay_ratio


def main(argv):

    learner = Learner()
    learner.train()
    return

if __name__ == '__main__':
    tf.app.run()
