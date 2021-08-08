import numpy as np


class kuhn_munkres(object):
    def __init__(self, graph: np.ndarray):
        self.transpose = (graph.shape[0] > graph.shape[1])
        graph = graph.transpose() if self.transpose else graph
        self.graph = graph
        self.n = graph.shape[0]
        self.m = graph.shape[1]
        self.lx = np.max(graph, axis=1)  # x的初始权重为所有边的最大值
        self.ly = np.empty(shape=self.m, dtype=graph.dtype)  # y的初始权重都为0
        self.ly.fill(0)
        self.visx = np.empty(shape=self.n, dtype=np.bool)
        self.visy = np.empty(shape=self.m, dtype=np.bool)
        self.match = np.empty(shape=self.m, dtype=np.int)
        self.match.fill(-1)
        self.slack = np.empty(shape=self.m, dtype=graph.dtype)

    def wrap_return(self):
        if self.transpose:
            idx = np.argsort(self.match, axis=0)
            return idx[-self.n:]
        else:
            return self.match

    def find(self, x_idx):
        self.visx[x_idx] = True

        for i in range(self.m):
            if self.visy[i]:
                continue
            else:
                gap = self.lx[x_idx] + self.ly[i] - self.graph[x_idx][i]
                if gap <= 1e-6:
                    self.visy[i] = True
                    if self.match[i] == -1 or self.find(self.match[i]):
                        self.match[i] = x_idx
                        return True
                else:
                    self.slack[i] = self.slack[i] if self.slack[i] < gap else gap

        return False

    def __call__(self):
        for i in range(self.n):
            self.slack.fill(9999999)
            while True:
                self.visx.fill(False)
                self.visy.fill(False)
                if self.find(i):
                    break
                d = np.min(self.slack[~self.visy])
                self.lx[self.visx] -= d
                self.ly[self.visy] += d

        return self.wrap_return()


def parallel_km(graph):
    km = kuhn_munkres(graph)
    print(graph.shape)
    return km()


if __name__ == '__main__':
    T = 10
    # np.random.seed(1)

    graph = np.load('e.npy')
    a = kuhn_munkres(graph)
    # graphs = []
    # for i in range(T):
    #     N = np.random.randint(3, 10, size=2)
    #     graphs.append(np.random.randint(low=1, high=200, size=(N[0], N[1])))
    # a = Parallel(n_jobs=2)(delayed(parallel_km)(graph) for graph in graphs)
    print(a())
