import pickle
import sys
import struct
import numpy as np

from collections import namedtuple

MSGit_Embedding = namedtuple('MSGit_Embedding', ['image', 'caption', 'embedding', 'ratings'])

MAGIC = b'MSGitEmb'
VERSION = 1

def write_string(str, output_file):
	encoded = str.encode()
	count = len(encoded)
	output_file.write(struct.pack('q', count))
	output_file.write(encoded)

def read_string(contents, cur_pos):
	(count,) = struct.unpack('q', contents[cur_pos:cur_pos+8])
	cur_pos += 8
	result = contents[cur_pos:cur_pos+count].decode()
	return result, 8 + count

def write_np_array(arr, output_file):
	itemsize = arr.itemsize
	if itemsize not in (4, 8):
		print(f"Unable to serialize: Unsupported itemsize: {itemsize}")
		return
	output_file.write(struct.pack('I', arr.itemsize))
	shape = arr.shape
	count = len(shape)
	output_file.write(struct.pack('I', count))
	for i in range(count):
		output_file.write(struct.pack('q', shape[i]))
	result_bytes = arr.tobytes()
	output_file.write(result_bytes)

def read_np_array(contents, cur_pos):
	(itemsize,) = struct.unpack('I', contents[cur_pos:cur_pos+4])
	if itemsize not in (4, 8):
		print(f"Unable to deserialize: Unsupported itemsize: {itemsize}")
		return None, len(contents)
	cur_pos += 4
	(count,) = struct.unpack('I', contents[cur_pos:cur_pos+4])
	cur_pos += 4
	shape = struct.unpack('q'*count, contents[cur_pos:cur_pos + 8*count])
	cur_pos += 8*count
	size = itemsize
	for dim in shape:
		size *= dim
	if itemsize == 4:
		dtype = np.single
	else:
		dtype = np.double
	result_data = np.frombuffer(contents[cur_pos:cur_pos+size], dtype=dtype)
	result = result_data.reshape(shape)
	return result, 4 + 4 + 8*count + size

def write_float_list(lst, output_file):
	output_file.write(struct.pack('q', len(lst)))
	for x in lst:
		output_file.write(struct.pack('d', x))

def read_float_list(contents, cur_pos):
	(count,) = struct.unpack('q', contents[cur_pos:cur_pos+8])
	cur_pos += 8
	result_data = struct.unpack('d'*count, contents[cur_pos:cur_pos + 8*count])
	result = list(result_data)
	return result, 8 + 8*count

#####################

def serialize(msgit_embeddings, output_file):
	output_file.write(MAGIC)
	output_file.write(struct.pack('I', VERSION))
	output_file.write(struct.pack('I', 0)) # Reserved
	count = len(msgit_embeddings)
	output_file.write(struct.pack('q', count))
	for ge in msgit_embeddings:
		write_string(ge.image, output_file)
		write_string(ge.caption, output_file)
		write_np_array(ge.embedding, output_file)
		write_float_list(ge.ratings, output_file)

def deserialize(input_file):
	contents = input_file.read()
	cur_pos = 0
	if contents[cur_pos:cur_pos + 8] != MAGIC:
		print(f"Invalid file contents for deserialize: Incorrect Magic Value: {contents[cur_pos:cur_pos + 8]}")
		return None
	cur_pos += 8
	(version,) = struct.unpack('I', contents[cur_pos:cur_pos + 4])
	if version > VERSION:
		print(f"Unable to deserialize: File is Version {version}, but deserialization routine is Version {VERSION}")
		return None
	cur_pos += 4
	cur_pos += 4 # Reserved
	(count,) = struct.unpack('q', contents[cur_pos:cur_pos + 8])
	cur_pos += 8
	result = []
	for i in range(count):
		image, num_bytes = read_string(contents, cur_pos)
		cur_pos += num_bytes
		caption, num_bytes = read_string(contents, cur_pos)
		cur_pos += num_bytes
		embedding, num_bytes = read_np_array(contents, cur_pos)
		cur_pos += num_bytes
		ratings, num_bytes = read_float_list(contents, cur_pos)
		cur_pos += num_bytes
		result.append(MSGit_Embedding(image, caption, embedding, ratings))
	return result

def lists_eq(a, b):
	if len(a) != len(b):
		return False
	for x, y in zip(a, b):
		if x != y:
			return False
	return True

def compare_ge(msgit_embeddings1, msgit_embeddings2):
	if len(msgit_embeddings1) != len(msgit_embeddings2):
		print("Different lengths!")
	for ge1, ge2 in zip(msgit_embeddings1, msgit_embeddings2):
		if ge1.image != ge2.image:
			print("Different images!")
		if ge1.caption != ge2.caption:
			print("Different captions!")
		if (ge1.embedding != ge2.embedding).any():
			print("Different embeddings!")
		if not lists_eq(ge1.ratings, ge2.ratings):
			print("Different ratings!")

def main():
	input_path = sys.argv[1]
	output_path = 'embeddings/' + '.'.join(input_path.split('/')[-1].split('.')[:-1]) + '.ge'

	print("Deserializing")
	with open(input_path, 'rb') as in_file:
		msgit_embeddings = deserialize(in_file)
	
	print("Serializing")
	with open(output_path, 'wb') as out_file:
		serialize(msgit_embeddings, out_file)
	
	print("Deserializing")
	with open(output_path, 'rb') as in_file:
		msgit_embeddings_deserialized = deserialize(in_file)
	
	compare_ge(msgit_embeddings, msgit_embeddings_deserialized)

	print("length:", len(msgit_embeddings))
	print("shape:", msgit_embeddings[0].embedding.shape)

if __name__ == "__main__":
	main()

