import mxnet as mx
from mxnet import nd


class AvgMetric(mx.metric.EvalMetric):
    def __init__(self, name='accuracy',
                 output_names=None, label_names=None):

        super(AvgMetric, self).__init__(name, output_names=output_names, label_names=label_names)
        self.value_list = []
        self.delimiters = []

    def reset(self):
        if hasattr(self,'delimiters'):
            self.delimiters.append(len(self.value_list))
        self.num_inst = 0
        self.sum_metric = 0.0

    def update(self, value, n=1):
        self.sum_metric += n * value
        self.num_inst += n
        self.value_list += [value]*n

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            res = (self.sum_metric / self.num_inst)
            if isinstance(res, nd.NDArray):
                res = res.asscalar()
            return (self.name, res)
