import random
import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    PRECISION = 5
    MIN_POSSIBLE_PRIORITY = 1e-5
    write = 0
    min_priority = 0xffffffff

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        p = round(float(p), SumTree.PRECISION)
        if p == 0:
            p = self.MIN_POSSIBLE_PRIORITY

        idx = self.write + self.capacity - 1
        self.pending_idx.add(idx)

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

        if self.min_priority > p:
            self.min_priority = p

    # update priority
    def update(self, idx, p):
        p = round(float(p), SumTree.PRECISION)
        if idx not in self.pending_idx:
            return
        self.pending_idx.remove(idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        self.pending_idx.add(idx)
        return idx, self.data[data_idx], self.tree[idx]


# TESTS
def test_1():
    result = True
    st = SumTree(5)
    for i in range(5):
        st.add(i + 1, i + 1)
    expected_total = 15
    if st.total() != expected_total:
        result = False

    expected_tree = [15, 10, 5, 9, 1, 2, 3, 4, 5]
    for i in range(9):
        if st.tree[i] != expected_tree[i]:
            result = False

    return result


def test_2_update():
    result = True
    st = SumTree(5)
    for i in range(5):
        st.add(i + 1, i + 1)
    expected_total = 15
    if st.total() != expected_total:
        result = False
    st.get(9.5)
    st.update(4, 100)
    expected_tree = [114, 109, 5, 9, 100, 2, 3, 4, 5]
    for i in range(9):
        if st.tree[i] != expected_tree[i]:
            result = False
    return result


def test_3_update():
    result = True
    st = SumTree(100000)
    for i in range(19705):
        st.add(1000, 1)

    expected_total = 1000 * 19705
    if st.total() != expected_total:
        result = False
    random.seed(1)
    for i in range(1000):
        s = random.random() * st.total()
        data = st.get(s)
        st.update(data[0], random.random())

    for i in range(len(st.tree) // 2 - 2):
        if round(st.tree[i] - (st.tree[2 * i + 1] + st.tree[2 * i + 2]), 5) != 0:
            result = False
            print("{0} <> {1} + {2} dif = {3}".format(st.tree[i], st.tree[2 * i + 1], st.tree[2 * i + 2],
                  st.tree[i] - (st.tree[2 * i + 1] + st.tree[2 * i + 2])))
    return result


def run_tests():
    if not test_1():
        print("Test 1 Failed")
    else:
        print("Test 1 OK")

    if not test_2_update():
        print("Test 2 Failed")
    else:
        print("Test 2 OK")

    if not test_3_update():
        print("Test 3 Failed")
    else:
        print("Test 3 OK")


if __name__ == '__main__':
    run_tests()
