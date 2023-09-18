import torch
import numpy as np
import timeit
import old_functions as old
from acvl_utils.cropping_and_padding.bounding_boxes import pad_bbox, regionprops_bbox_to_proper_bbox, bounding_box_to_slice, get_bbox_from_mask, get_bbox_from_mask_npwhere
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from acvl_utils.instance_segmentation.instance_as_semantic_seg import _internal_convert_semantic_to_instance_mp, convert_semantic_to_instanceseg
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component, generate_ball, generic_filter_components

def test(func,multiplier_ratio,*args):
    global results,labels,index,multiplier
    n=int(multiplier/multiplier_ratio)
    results[index]=np.average(timeit.repeat(lambda:func(*args), number=n))
    labels.append(func.__name__)
    index+=1
    func_old=getattr(old,func.__name__)
    results[index]=np.average(timeit.repeat(lambda:func_old(*args), number=n))
    labels.append("old."+func.__name__)
    index+=1

results=np.zeros(100)
labels=[]
index=0
multiplier=10000

bbox = [[32, 64], [21, 46]]
mask=np.ndarray(shape=(2,3,2))
torch_tensor_2d = torch.rand((32, 23))
torch_tensor_5d = torch.rand((1, 3, 57, 18, 10))
new_shape = (96, 64)
segmentation_mask=np.random.randint(0,64,size=(300,200),dtype='uint8')
sample = [True, False]
binary_image = np.random.choice(sample, size=(300,200))

test(pad_bbox,1,bbox,3)
test(regionprops_bbox_to_proper_bbox,0.1,bbox)
test(bounding_box_to_slice,0.1,bbox)

test(get_bbox_from_mask,2,mask)
test(get_bbox_from_mask_npwhere,2,mask)

test(pad_nd_image,5,torch_tensor_2d,new_shape)
test(pad_nd_image,30,torch_tensor_5d,new_shape)

#TODO _internal_convert_semantic_to_instance_mp
test(convert_semantic_to_instanceseg,100,segmentation_mask)
test(generate_ball,50,(10,10,10),(5,5,5))
test(remove_all_but_largest_component,500,binary_image)
test(generic_filter_components,500,binary_image,old.filter_fn)

results=np.trim_zeros(results)
labels=np.trim_zeros(labels)
data=' '.join(labels)+'\n'+' '.join(str(i) for i in results)+'\n'+str(multiplier)
with open("results.txt", 'w') as f:
    f.write(data)