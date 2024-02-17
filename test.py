list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

zipped = zip(list1, list2)

# The result is an iterator of tuples
# [(1, 'a'), (2, 'b'), (3, 'c')]
print(zipped)


import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
else:
    print("GPU is not available.")

