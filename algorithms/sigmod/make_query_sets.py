import numpy as np

queries_path = "data/contest-queries-release-1m.bin"
type_1_fp = "data/type_1_queries_1m1m.bin"
type_2_fp = "data/type_2_queries_1m1m.bin"
type_3_fp = "data/type_3_queries_1m1m.bin"
type_4_fp = "data/type_4_queries_1m1m.bin"

data_size = 0
with open(queries_path, 'rb') as file:
    size = file.read(4)
    data_size = np.frombuffer(size, dtype=np.int32)[0]
    print(data_size)

# Each vector has the following structure:
dtype_vector = np.dtype([
    ('category', np.float32),
    ('v', np.float32),
    ('l', np.float32),
    ('r', np.float32),
    ('dims', np.float32, 100)
])
# The dims have 100 floats, all indexable.

# Read all the query vectors
data = np.fromfile(queries_path, dtype=dtype_vector, offset=4)


query_sets = [[] for _ in range(4)]

for vector in data:
    category = int(vector['category'])
    query_sets[category].append(vector)

for q in query_sets:
    print(len(q))

with open(type_1_fp, 'wb') as file:
    length = len(query_sets[0])
    file.write(length.to_bytes(4, byteorder='little'))
    for q in query_sets[0]:
        vec = q['category'].tobytes() + q['v'].tobytes() + q['l'].tobytes() + \
              q['r'].tobytes() + q['dims'].tobytes()
        file.write(vec)

with open(type_2_fp, 'ab') as file:
    length = len(query_sets[1])
    file.write(length.to_bytes(4, byteorder='little'))
    for q in query_sets[1]:
        vec = q['category'].tobytes() + q['v'].tobytes() + q['l'].tobytes() + \
              q['r'].tobytes() + q['dims'].tobytes()
        file.write(vec)

with open(type_3_fp, 'ab') as file:
    length = len(query_sets[2])
    file.write(length.to_bytes(4, byteorder='little'))
    for q in query_sets[2]:
        vec = q['category'].tobytes() + q['v'].tobytes() + q['l'].tobytes() + \
              q['r'].tobytes() + q['dims'].tobytes()
        file.write(vec)

with open(type_4_fp, 'ab') as file:
    length = len(query_sets[3])
    file.write(length.to_bytes(4, byteorder='little'))
    for q in query_sets[0]:
        vec = q['category'].tobytes() + q['v'].tobytes() + q['l'].tobytes() + \
              q['r'].tobytes() + q['dims'].tobytes()
        file.write(vec)


print("Shape:", data.shape)
print(data[0]['dims'][99])
print("a")
