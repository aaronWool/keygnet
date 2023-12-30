import os 
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import platform
import open3d as o3d

def rgbd_to_point_cloud(K, depth,rgb):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    #print(zs.min())
    #print(zs.max())
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs,rgb[vs,us,0],rgb[vs,us,1],rgb[vs,us,2]]).T
    return pts

class BOPDataset(Dataset):
    def __init__(self,root,set,min_visible_points=2000,points_count_net = 1024) -> None:
        super().__init__()
        self.root = root
        self.set=set
        self.points_count_net = points_count_net
        self.cycle_path = os.path.join(root,set)
        self.split_path = os.path.join(self.root,"split")
        #standarization of ImageNet
        self.mean = np.array([0.485, 0.456, 0.406],dtype=np.float64)
        self.std = np.array([0.229, 0.224, 0.225],dtype=np.float64)

        #generate splits
        if not os.path.exists(self.split_path):
            os.mkdir(self.split_path)
        if not os.path.isfile(os.path.join(self.split_path,set+'.txt')):
            print("Split txt not exists, generating ", self.set, " split file...")
            split_txt = open(os.path.join(self.split_path,set+'.txt'),'w')
            for cycle in os.listdir(self.cycle_path):
                if os.path.isdir(os.path.join(self.cycle_path,cycle)):
                    #print(cycle)
                    #scene_gt_info_json = open(os.path.join(self.cycle_path, cycle, 'scene_gt_info.json'),'r')
                    #scene_gt_info = json.load(scene_gt_info_json)
                    #for scene in scene_gt_info:
                    #    split_txt.writelines(os.path.join(cycle,str(scene).zfill(6))+'\n')
                    for mask_name in os.listdir(os.path.join(self.cycle_path,cycle,'mask_visib')):
                        mask = np.array(Image.open(os.path.join(self.cycle_path,cycle,'mask_visib',mask_name)))
                        #print(mask.shape)
                        #plt.imshow(mask)
                        #plt.show()
                        #print(np.count_nonzero(mask))
                        if np.count_nonzero(mask)>=min_visible_points:
                            split_txt.writelines(cycle+'/'+str(mask_name.split('.')[0]).zfill(6)+'\n')
            split_txt.close()
        with open(os.path.join(self.split_path, self.set+".txt")) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]
        #generate normalization parameters
        self.cad_models_path = os.path.join(self.root,"models")
        cad_models_info_json = open(os.path.join(self.cad_models_path,'models_info.json'),'r')
        cad_models_info = json.load(cad_models_info_json)
        #x, y, z
        self.coor_dims = np.zeros(3)
        for i in cad_models_info:
            if self.coor_dims[0]<cad_models_info[i]['size_x']/2:
                self.coor_dims[0]=cad_models_info[i]['size_x']/2
            if self.coor_dims[1]<cad_models_info[i]['size_y']/2:
                self.coor_dims[1]=cad_models_info[i]['size_y']/2
            if self.coor_dims[2]<cad_models_info[i]['size_z']/2:
                self.coor_dims[2]=cad_models_info[i]['size_z']/2
        #dump normalization parameters
        np.savetxt(os.path.join(self.split_path,'coor_dims.txt'),self.coor_dims)
        #print(self.coor_dims)


    def transform(self,rgb,depth,cam_k):
        #rgb -= self.mean
        #rgb /= self.std
        pts=rgbd_to_point_cloud(cam_k,depth,rgb)
        #print(pts)
        #np.savetxt('a.txt',pts)
        #recenter and normalize pc

        #remove outlier noise of the visib_mask
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:,0:3])
        pcd.colors = o3d.utility.Vector3dVector(pts[:,3:])
        #pcd.remove_statistical_outlier(100,0.01)
        #cl,idx = pcd.remove_statistical_outlier(20,2.0)
        cl,idx=pcd.remove_radius_outlier(nb_points=15, radius=20,print_progress=False)
        #print(idx)
        pcd = pcd.select_by_index(idx)
        #pcd = pcd.farthest_point_down_sample(self.points_count_net)
        #o3d.visualization.draw_geometries([pcd])
        pts_tmp = np.asarray(pcd.points)
        pts  = np.zeros((pts_tmp.shape[0],6))
        pts[:,0:3]=pts_tmp
        pts[:,3:] = np.asarray(pcd.colors)
        if pts.shape[0]<self.points_count_net:
            pts = np.concatenate((pts,np.zeros((self.points_count_net-pts.shape[0],6))),axis=0)
            #print(pts.shape)
        else:
            idx=np.random.choice(np.arange(pts.shape[0]),self.points_count_net,replace=False)
            pts = pts[idx]
        for i in range(3):
            #print(i)
            pts[:,i] -= np.mean(pts[:,i])
            pts[:,i] /=self.coor_dims[i]
        pts = torch.from_numpy(pts).float()
        return pts

    def __getitem__(self, index):
        id = self.ids[index]
        cycle, scene_objidx = id.split('/')
        scene, objidx = scene_objidx.split('_')
        if self.set == 'train':
            rgb = np.array(Image.open(os.path.join(self.cycle_path,cycle,'rgb',scene+'.jpg')))
        else:
            rgb = np.array(Image.open(os.path.join(self.cycle_path,cycle,'rgb',scene+'.png')))
        rgb = rgb.astype('float64')
        rgb/=255.
        #plt.imshow(rgb)
        #plt.show()
        
        depth = np.array(Image.open(os.path.join(self.cycle_path,cycle,'depth',scene+'.png')))
        mask_visb = np.array(Image.open(os.path.join(self.cycle_path,cycle,'mask_visib',scene_objidx+'.png')))
        cam_k_json = json.load(open(os.path.join(self.cycle_path, cycle, 'scene_camera.json'),'r'))
        #print(cam_k_json['0'])
        cam_k = np.array(cam_k_json[str(int(scene))]['cam_K'])
        cam_k = cam_k.reshape(3,3)
        depth_scale = cam_k_json[str(int(scene))]['depth_scale']
        depth = depth*depth_scale
        depth = np.where(mask_visb==255,depth,0)
        #print(cam_k)
        pts=self.transform(rgb,depth,cam_k)
        #pts=rgbd_to_point_cloud(cam_k,depth,rgb)
        #np.savetxt('a.txt',pts)

        return pts
    def __len__(self):
        return len(self.ids)
if __name__ == "__main__":
    from torch.utils import data
    import matplotlib.pyplot as plt
    import os
    root = 'D:/Datasets/6dPoseData/lm'
    set = 'train'
    test_loader=data.DataLoader(BOPDataset(root,set),batch_size=8,shuffle=False)
    for bacth_id,pts in enumerate(test_loader):
        print(pts.shape)
        #np.savetxt('a.txt',pts[0])
        #os.system('pause')
        #plt.imshow(depth[0])
        #plt.show()