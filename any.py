import sys
import tensorflow as tf

v_list = []
for e in tf.train.summary_iterator(sys.argv[1]):
    for v in e.summary.value:
        if v.tag == sys.argv[2]:
            v_list.append(v.simple_value)

predicate = eval(sys.argv[3])
print(any(map(predicate, v_list)))
