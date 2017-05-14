import numpy as np
import os
import math
import matplotlib.pyplot as plt
import skimage.io
from skimage.color import rgb2gray
from skimage.filters import median, threshold_otsu
from skimage.morphology import skeletonize_3d
from skimage.measure import label
from scipy.signal import convolve2d
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import networkx as nx

def LoadImage(path, filename):
	image = skimage.io.imread(os.path.join(path,filename), 0)
	return image

def PreprocessImage(img, image_name):
	grey = rgb2gray(img)
	th = threshold_otsu(grey)
	return grey < th

def CreateSkeleton(img, image_name):
	return skeletonize_3d(img)
#too slow
def RozenfeldSceletonize(img, image_name):
	skel = img.astype(int)
	kernel = np.zeros((3,3))
	kernel[0,0] = 0b00000001
	kernel[0,1] = 0b00000010
	kernel[0,2] = 0b00000100
	kernel[1,2] = 0b00001000
	kernel[2,2] = 0b00010000
	kernel[2,1] = 0b00100000
	kernel[2,0] = 0b01000000
	kernel[1,0] = 0b10000000
	def function(mask):
		mask = np.array([int(i) for i in bin(int(mask))[2:].zfill(9)])
		tomap = {'04':[[1,2,3],[5,6,7]],'15':[[2,3,4],[0,6,7]],'26':[[0,1,7],[3,4,5]],\
				'37':[[0,1,2],[4,5,6]],'13':[[0,4,5,6,7],[2]],'35':[[0,1,2,6,7],[4]],\
				'17':[[2,3,4,5,6],[0]],'57':[[0,1,2,3,4],[6]]}
		for diag in sorted(tomap.keys()):
			if mask[int(diag[0])]+mask[int(diag[1])]==0 \
				and np.sum(mask[tomap[diag][0]])!=0 and np.sum(mask[tomap[diag][1]])!=0:
				return 0
		return 1
	def create_8simple(pic):
		simp_mask = convolve2d(pic, kernel, mode='same')
		simp = np.zeros(pic.shape)
		#simp = map(lambda i: function(i), simp_mask)
		squarer = lambda i: function(i)
		vfunc = np.vectorize(squarer)
		simp = vfunc(simp_mask)
		simp = simp==1
		print(np.array(simp).shape)
		#plt.imshow(simp)
		#plt.show()
		return simp
	def create_8isolated(pic):
		isol = pic+np.roll(pic,1,axis=0)+np.roll(pic,-1,axis=0)+np.roll(pic,1,axis=1)+np.roll(pic,-1,axis=1)\
			+np.roll(np.roll(pic,1,axis=0),1,axis=1)+np.roll(np.roll(pic,1,axis=0),-1,axis=1)\
			+np.roll(np.roll(pic,-1,axis=0),1,axis=1)+np.roll(np.roll(pic,-1,axis=0),-1,axis=1)
		isol = isol==1
		return isol
	def create_8endpoint(pic):
		endp = pic-np.roll(pic,1,axis=0)-np.roll(pic,-1,axis=0)-np.roll(pic,1,axis=1)-np.roll(pic,-1,axis=1)\
			-np.roll(np.roll(pic,1,axis=0),1,axis=1)-np.roll(np.roll(pic,1,axis=0),-1,axis=1)\
			-np.roll(np.roll(pic,-1,axis=0),1,axis=1)-np.roll(np.roll(pic,-1,axis=0),-1,axis=1)
		endp = ((endp==0)+(endp==1))*(pic!=0)
		return endp
	def check_east(pic):
		north = pic-np.roll(pic,1,axis=1)
		north = north==1
		return north
	def check_west(pic):
		south = pic-np.roll(pic,-1,axis=1)
		south = south==1
		return south
	def check_south(pic):
		east = pic-np.roll(pic,-1,axis=0)
		east = east==1
		return east
	def check_north(pic):
		west = pic-np.roll(pic,1,axis=0)
		west = west==1
		return west
	n_changed_pixels = 1
	while n_changed_pixels > 0:
		prev_skel = skel.copy()
		skel1 = check_north(skel)*create_8simple(skel)*np.invert(create_8endpoint(skel))*np.invert(create_8isolated(skel))
		skel2 = check_south(skel)*create_8simple(skel)*np.invert(create_8endpoint(skel))*np.invert(create_8isolated(skel))
		skel3 = check_east(skel)*create_8simple(skel)*np.invert(create_8endpoint(skel))*np.invert(create_8isolated(skel))
		skel4 = check_west(skel)*create_8simple(skel)*np.invert(create_8endpoint(skel))*np.invert(create_8isolated(skel))
		skel = skel1+skel2+skel3+skel4
		n_changed_pixels = np.sum(prev_skel-skel)
		
		fig, ((ax1,ax2,ax3),(ax4,ax5, ax6)) = plt.subplots(2,3)
		ax1.imshow(prev_skel)
		ax2.imshow(skel1)
		ax3.imshow(skel2)
		ax4.imshow(skel3)
		ax5.imshow(skel4)
		ax6.imshow(skel)
		plt.title(str(n_changed_pixels)+'changed')
		plt.show()
	return skel

def CreateGraph(skel_orig, image_name):
	skel = skel_orig // 255
	new_skel = skel.astype(int)
	coordinates = []
	G=nx.Graph()
	vertice = 0

	for x in range(skel.shape[0]):
		for y in range(skel.shape[1]):
			if not skel[x,y]:
				continue
			new_skel[x,y] = skel[x-1,y-1]+skel[x,y-1]+skel[x+1,y-1]+\
							skel[x-1,y]+skel[x+1,y]+\
							skel[x-1,y+1]+skel[x,y+1]+skel[x+1,y+1]
			if new_skel[x,y] != 2:
				G.add_node(vertice, pos=(x,y))
				coordinates.append([x,y])
				vertice += 1
	di = {'0':[-1,-1],'1':[0,-1],'2':[1,-1],'3':[1,0],'4':[1,1],'5':[0,1],'6':[-1,1],'7':[-1,0]}
	def search_node(coordinates, img, pos):
		edge_length = 0
		new_nodes = []
		nodes_length = []
		for mama_dir in sorted(di.keys()):
			#print(pos[0], pos[1])
			if not img[pos[0]+di[mama_dir][0],pos[1]+di[mama_dir][1]]:
				continue
			else:
				prev_x = pos[0]
				prev_y = pos[1]
				cur_x = prev_x + di[mama_dir][0]
				cur_y = prev_y + di[mama_dir][1]
				if [cur_x,cur_y] in coordinates:
					new_nodes += [coordinates.index([cur_x,cur_y])]
					nodes_length += [edge_length]
					#print("LENGTH 0")
					continue
				while True:
					#print('>>>WHILE<<<')
					#print(coordinates)
					found = False
					for n_dir in sorted(di.keys()):
						#print('~'+n_dir+'~')
						#print(prev_x, prev_y, cur_x, cur_y, cur_x+di[n_dir][0], cur_y+di[n_dir][1])
						if not img[cur_x+di[n_dir][0],cur_y+di[n_dir][1]]:
							#print('->white')
							continue
						if cur_x+di[n_dir][0]==prev_x and cur_y+di[n_dir][1]==prev_y:
							#print('->prev')
							continue
						prev_x = cur_x
						prev_y = cur_y
						cur_x += di[n_dir][0]
						cur_y += di[n_dir][1]
						edge_length += 1
						if [cur_x,cur_y] in coordinates:
							new_nodes += [coordinates.index([cur_x,cur_y])]
							nodes_length += [edge_length]
							edge_length = 0
							found = True
						break
					if found:
						break
					#print('OOOOHHHHH STH WRONG')
		return new_nodes, nodes_length
	
	G_copy = G.copy()
	for node_num, coords in G_copy.nodes(data=True):
		new_nodes, edges_length = search_node(coordinates, skel, coords['pos'])
		for n in range(len(new_nodes)):
			G.add_edge(node_num, new_nodes[n], len=edges_length[n])
	
	def check_dist(coords, popp, x, y, idx):
		for c_id, c in enumerate(coords[idx:]):
			if idx+c_id in popp:
				continue
			if math.hypot(c[0] - x, c[1] - y) != 0 and math.hypot(c[0] - x, c[1] - y) < 3:
				return idx+c_id
		return -1
	
	def remove_close_nodes(G, popped, vertice, coordinates):
		while True:
			merged = False
			for vert1, attr in G.nodes(data=True):
				vert2 = check_dist(coordinates, popped, attr['pos'][0], attr['pos'][1], vert1+1)
				if vert2 != -1:
					vertice += 1
					merge_nodes(G, [vert1, vert2], vertice, {'pos':(coordinates[vert1][0], coordinates[vert1][1])})
					coordinates += [coordinates[vert1]]
					popped += [vert1, vert2]
					merged = True
				if merged:
					break
			if not merged:
				break
		return popped, vertice, coordinates
	#not used
	def remove_extra_terminal_nodes(G, popped, vertice):
		G_copy = G.copy()
		for (edge1, edge2, attr) in G_copy.edges_iter(data=True):
			if (edge1==edge2 and attr['len'] < 5) or (attr['len'] < 5 and (G.degree(edge1)==1 or G.degree(edge2)==1)):
				if edge1==edge2:
					G.remove_edge(edge1, edge2)
				if G.degree(edge1) == 1:
					G.remove_node(edge1)
					popped += [edge1]
				elif G.degree(edge2) == 1:
					G.remove_node(edge2)
					popped += [edge2]
		return popped, vertice
	def remove_extra_nodes(G, popped, vertice, coordinates):
		while True:
			merged = False
			for node, attr in G.nodes(data=True):
				if G.degree(node) == 2:
					n1 = G.neighbors(node)[0]
					n2 = G.neighbors(node)[1]
					G.add_edge(n1, n2, len=G.get_edge_data(node,n1)['len']+G.get_edge_data(node,n2)['len'])
					G.remove_node(node)
					popped += [node]
					merged = True
				elif G.degree(node) == 3:
					n1 = G.neighbors(node)[0]
					n2 = G.neighbors(node)[1]
					n3 = G.neighbors(node)[2]
					lengths = [G.get_edge_data(node,n1)['len'], G.get_edge_data(node,n2)['len'], G.get_edge_data(node,n3)['len']]
					length_m = [(l<=7) for l in lengths]
					lengths_l = [l>50 for l in lengths]
					if sum(length_m)==1 and sum(lengths_l)>0 and G.degree([n for a,n in zip(length_m,[n1,n2,n3]) if a][0]) == 1:
						nodes = [n for a,n in zip(length_m,[n1,n2,n3]) if not a]
						G.add_edge(nodes[0], nodes[1], len=G.get_edge_data(node,nodes[0])['len']+G.get_edge_data(node,nodes[1])['len'])
						G.remove_node(node)
						#pos = dict(zip(range(len(coordinates)), coordinates))
						#plt.imshow(new_skel.T)
						#nx.draw_networkx(G, pos, node_size=5)
						#plt.show()
						popped += [node]
						merged = True
				if merged:
					break
			if not merged:
				break
		return popped, vertice
	def remove_self_loops(G):
		G_copy = G.copy()
		for (edge1, edge2, attr) in G_copy.edges_iter(data=True):
			if edge1==edge2:
				G.remove_edge(edge1, edge2)
	def remove_isolated_nodes(G, popped):
		G_copy = G.copy()
		for node, attr in G_copy.nodes(data=True):
			if G.degree(node) == 0:
				G.remove_node(node)
				popped += [node]
		return popped

	pos = dict(zip(range(len(coordinates)), coordinates))
	#fig, ax = plt.subplots(figsize=(20,10))
	#ax.imshow(new_skel.T)
	#nx.draw_networkx(G, pos, node_size=5)
	#plt.show()
	#fig.savefig('Result/'+image_name[:-4]+'_skel_1.png')
	
	popped = []
	vertice -= 1
	popped, vertice, coordinates = remove_close_nodes(G, popped, vertice, coordinates)
	remove_self_loops(G)
	popped, vertice = remove_extra_nodes(G, popped, vertice, coordinates)
	popped = remove_isolated_nodes(G, popped)

	pos = dict(zip(range(len(coordinates)), coordinates))
	#fig, ax = plt.subplots(figsize=(20,10))
	#ax.imshow(new_skel.T)
	#nx.draw_networkx(G, pos, node_size=5)
	#plt.title(image_name)
	#plt.show()
	#fig.savefig('Result/'+image_name[:-4]+'_skel_2.png')
	
	#to_save = np.dstack((skel_orig,skel_orig,skel_orig))
	#for node, attr in G.nodes(data=True):
	#	to_save[attr['pos'][0],attr['pos'][1]] = (0,0,255)
	#skimage.io.imsave('Result/'+image_name+'.png',to_save)
	return G, new_skel

def merge_nodes(G, nodes, new_node, attr_dict=None, **attr):
	G.add_node(new_node, attr_dict, **attr)
	for n1,n2,data in G.edges(data=True):
		if n1 in nodes:
			G.add_edge(new_node,n2,data)
		elif n2 in nodes:
			G.add_edge(n1,new_node,data)
	for n in nodes:
		G.remove_node(n)

def FindFeatures(G, skel, long_th):
	number_components = nx.number_connected_components(G)
	number_nodes = G.number_of_nodes()
	number_edges = G.number_of_edges()
	edge_lens = []
	degrees = []
	terminal_nodes = []
	for (edge1, edge2, attr) in G.edges_iter(data=True):
		edge_lens += [attr['len']]
	for node, attr in G.nodes(data=True):
		degrees += [G.degree(node)]
		if G.degree(node) == 1:
			if terminal_nodes == []:
				terminal_nodes += [node]
				continue
			dist = [math.hypot(attr['pos'][0] - nx.get_node_attributes(G,'pos')[node2][0], attr['pos'][1] - nx.get_node_attributes(G,'pos')[node2][1]) > 5 for node2 in terminal_nodes if node != node2]
			if all(dist):
				terminal_nodes += [node]
			#else:
			#	print(dist)
			#	print(terminal_nodes)
			#	input()
	#last_nodes = len([x for x in G.nodes_iter() if G.degree(x)==1])
	max_degree = max(degrees)
	count_max_degrees = np.sum(np.asarray(degrees) > 3)
	count_long_edges = np.sum(np.asarray(edge_lens) > long_th)
	#max_len = max(edge_lens)
	#avg_edge_len = np.mean(edge_lens)
	
	to_save = np.dstack((skel,skel,skel))
	for node, attr in G.nodes(data=True):
		if node in terminal_nodes:
			to_save[attr['pos'][0],attr['pos'][1]] = (0,255,0)
		else:
			to_save[attr['pos'][0],attr['pos'][1]] = (0,0,255)
	skimage.io.imsave('Result/'+image_name+'.png',to_save)
	return [number_components*10, len(terminal_nodes), max_degree, count_long_edges]

def create_gt(names):
	y = []
	for name in names:
		y += [int(name[7])]
	return y

def classify(x, y, names, lst):
	clf = KNeighborsClassifier(n_neighbors=1)
	#clf.fit(x[::4],y[::4])
	clf.fit([x[i] for i in lst], [y[i] for i in lst])
	print('>>',np.sum(np.asarray(y) == clf.predict(x))/len(y))
	#print('+>',np.asarray(y) == clf.predict(x))
	#[print(names[i], y[i], clf.predict(x[i:i+1]), x[i]) for i in range(len(names))]
	return np.sum(np.asarray(y) == clf.predict(x))/len(y)

if __name__ == '__main__':
	path = 'Geogliph_1'
	out_path = 'Result'
	list_train_images = [0,4,8,12,16,20,24]
	if not os.path.exists(out_path):
		os.makedirs(out_path)
	images = os.listdir(path)
	all_features = []
	images_names = []
	for image_name in images:
		if image_name.startswith("."):
			continue
		#if image_name != 'Силуэт_3_4.bmp':
		#	continue
		img = LoadImage(path, image_name)
		binarized = PreprocessImage(img, image_name)
		skeleton = CreateSkeleton(binarized, image_name)
		#skeleton = RozenfeldSceletonize(binarized, image_name)
		nwgraph, skel = CreateGraph(skeleton, image_name)
		all_features += [FindFeatures(nwgraph, skeleton, 30)]
		#fig, (ax1, ax2) = plt.subplots(1,2)
		#ax1.imshow(skeleton, interpolation = 'none', cmap = 'gray')
		#ax2.imshow(skel, interpolation = 'none')
		#plt.show()
		#fig.savefig(os.path.join(out_path,image_name[:-3]+'png'))
		images_names += [image_name]
	y = create_gt(images_names)
	classify(all_features,y, images_names, list_train_images)
#scores = [[0.6428571428571429, 0.6428571428571429, 0.6785714285714286, \
#0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, \
#0.6071428571428571, 0.6428571428571429, 0.6071428571428571, 0.6428571428571429, \
#0.6785714285714286, 0.6071428571428571, 0.6428571428571429, 0.5714285714285714, \
#0.5714285714285714, 0.6785714285714286, 0.7142857142857143, 0.6785714285714286, \
#0.6071428571428571, 0.6428571428571429, 0.6785714285714286, 0.7142857142857143, \
#0.7142857142857143, 0.8928571428571429, 0.8928571428571429, 0.8214285714285714, \
#0.9285714285714286, 0.9285714285714286, 0.8928571428571429, 0.9285714285714286, \
#0.9285714285714286, 0.9285714285714286, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,\
# 0.75, 0.75, 0.75, 0.75, 0.75, 0.7857142857142857, 0.8214285714285714, \
#0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.6785714285714286, \
#0.7142857142857143, 0.7142857142857143, 0.6785714285714286, 0.6428571428571429, \
#0.6428571428571429, 0.6428571428571429, 0.6071428571428571, 0.5714285714285714, \
#0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, \
#0.6785714285714286, 0.6785714285714286, 0.6785714285714286, 0.6785714285714286, \
#0.6785714285714286, 0.6785714285714286, 0.6785714285714286, 0.6785714285714286, \
#0.6785714285714286, 0.7142857142857143, 0.75, 0.75, 0.7857142857142857, \
#0.7142857142857143, 0.7142857142857143, 0.7142857142857143, 0.7142857142857143, \
#0.7142857142857143, 0.7142857142857143, 0.6785714285714286, 0.6785714285714286, \
#0.7142857142857143, 0.75, 0.75, 0.75, 0.7857142857142857, 0.7857142857142857, \
#0.75, 0.7857142857142857, 0.7857142857142857, 0.7142857142857143, 0.6428571428571429, \
#0.6428571428571429, 0.6785714285714286, 0.6785714285714286, 0.6071428571428571, 0.6428571428571429]]