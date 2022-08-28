import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# See https://github.com/ry/tensorflow-resnet/blob/master/resnet_train.py
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 16, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


class Config:
    def __init__(self):
        root = self.Scope('')
        # TODO: disabled the for-loop here.
        #  It causes an error absl.flags Error: Trying to access flag before flags were parsed when testing this script
        print(FLAGS.batch_size)  # For testing
        # for key in dir(FLAGS):
        #     root[key] = FLAGS.__getattr__(key)
        self.stack = [root]

    def items(self):
        return self.to_dict().items()

    def to_dict(self):
        self._pop_stale()
        out = {}
        # Work backwards from the flags to top of the stack
        # overwriting keys that were found earlier.
        for i in range(len(self.stack)):
            cs = self.stack[-i]
            for name in cs:
                out[name] = cs[name]
        return out

    def _pop_stale(self):
        # var_scope_name = tf.get_variable_scope().name
        var_scope_name = tf.compat.v1.get_variable_scope().name
        top = self.stack[0]
        while not top.contains(var_scope_name):
            # We aren't in this scope anymore
            self.stack.pop(0)
            top = self.stack[0]

    def __getitem__(self, name):
        self._pop_stale()
        # Recursively extract value
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return cs[name]

        raise KeyError(name)

    def set_default(self, name, value):
        if name not in self:
            self[name] = value

    def __contains__(self, name):
        self._pop_stale()
        for i in range(len(self.stack)):
            cs = self.stack[i]
            if name in cs:
                return True
        return False

    def __setitem__(self, name, value):
        self._pop_stale()
        top = self.stack[0]
        # var_scope_name = tf.get_variable_scope().name
        var_scope_name = tf.compat.v1.get_variable_scope().name
        assert top.contains(var_scope_name)

        if top.name != var_scope_name:
            top = self.Scope(var_scope_name)
            self.stack.insert(0, top)

        top[name] = value

    class Scope(dict):
        def __init__(self, name):
            self.name = name

        def contains(self, var_scope_name):
            return var_scope_name.startswith(self.name)


# Test
if __name__ == '__main__':

    def assert_raises(exception, fn):
        try:
            fn()
        except exception:
            pass
        else:
            assert False, 'Expected exception'

    c = Config()

    c['hello'] = 1
    assert c['hello'] == 1

    # with tf.variable_scope('foo'):
    with tf.compat.v1.variable_scope('foo'):
        c.set_default('bar', 10)
        c['bar'] = 2
        assert c['bar'] == 2
        assert c['hello'] == 1

        c.set_default('mario', True)

        # with tf.variable_scope('meow'):
        with tf.compat.v1.variable_scope('meow'):
            c['dog'] = 3
            assert c['dog'] == 3
            assert c['bar'] == 2
            assert c['hello'] == 1

            assert c['mario']

        assert_raises(KeyError, lambda: c['dog'])
        assert c['bar'] == 2
        assert c['hello'] == 1
        print("TESTING DONE")
