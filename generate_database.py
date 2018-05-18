import sys
import os
import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters

sys.path.append('./motion')

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from Learning import RBF

""" Options """

rng = np.random.RandomState(1234)
to_meters = 5.6444
window = 60
njoints = 31

""" Data """

data_terrain = [
    './data/animations/LocomotionFlat01_000.bvh',
    './data/animations/LocomotionFlat02_000.bvh',
    './data/animations/LocomotionFlat02_001.bvh',
    './data/animations/LocomotionFlat03_000.bvh',
    './data/animations/LocomotionFlat04_000.bvh',
    './data/animations/LocomotionFlat05_000.bvh',
    './data/animations/LocomotionFlat06_000.bvh',
    './data/animations/LocomotionFlat06_001.bvh',
    './data/animations/LocomotionFlat07_000.bvh',
    './data/animations/LocomotionFlat08_000.bvh',
    './data/animations/LocomotionFlat08_001.bvh',
    './data/animations/LocomotionFlat09_000.bvh',
    './data/animations/LocomotionFlat10_000.bvh',
    './data/animations/LocomotionFlat11_000.bvh',
    './data/animations/LocomotionFlat12_000.bvh',

    './data/animations/LocomotionFlat01_000_mirror.bvh',
    './data/animations/LocomotionFlat02_000_mirror.bvh',
    './data/animations/LocomotionFlat02_001_mirror.bvh',
    './data/animations/LocomotionFlat03_000_mirror.bvh',
    './data/animations/LocomotionFlat04_000_mirror.bvh',
    './data/animations/LocomotionFlat05_000_mirror.bvh',
    './data/animations/LocomotionFlat06_000_mirror.bvh',
    './data/animations/LocomotionFlat06_001_mirror.bvh',
    './data/animations/LocomotionFlat07_000_mirror.bvh',
    './data/animations/LocomotionFlat08_000_mirror.bvh',
    './data/animations/LocomotionFlat08_001_mirror.bvh',
    './data/animations/LocomotionFlat09_000_mirror.bvh',
    './data/animations/LocomotionFlat10_000_mirror.bvh',
    './data/animations/LocomotionFlat11_000_mirror.bvh',
    './data/animations/LocomotionFlat12_000_mirror.bvh',

    './data/animations/WalkingUpSteps01_000.bvh',
    './data/animations/WalkingUpSteps02_000.bvh',
    './data/animations/WalkingUpSteps03_000.bvh',
    './data/animations/WalkingUpSteps04_000.bvh',
    './data/animations/WalkingUpSteps04_001.bvh',
    './data/animations/WalkingUpSteps05_000.bvh',
    './data/animations/WalkingUpSteps06_000.bvh',
    './data/animations/WalkingUpSteps07_000.bvh',
    './data/animations/WalkingUpSteps08_000.bvh',
    './data/animations/WalkingUpSteps09_000.bvh',
    './data/animations/WalkingUpSteps10_000.bvh',
    './data/animations/WalkingUpSteps11_000.bvh',
    './data/animations/WalkingUpSteps12_000.bvh',

    './data/animations/WalkingUpSteps01_000_mirror.bvh',
    './data/animations/WalkingUpSteps02_000_mirror.bvh',
    './data/animations/WalkingUpSteps03_000_mirror.bvh',
    './data/animations/WalkingUpSteps04_000_mirror.bvh',
    './data/animations/WalkingUpSteps04_001_mirror.bvh',
    './data/animations/WalkingUpSteps05_000_mirror.bvh',
    './data/animations/WalkingUpSteps06_000_mirror.bvh',
    './data/animations/WalkingUpSteps07_000_mirror.bvh',
    './data/animations/WalkingUpSteps08_000_mirror.bvh',
    './data/animations/WalkingUpSteps09_000_mirror.bvh',
    './data/animations/WalkingUpSteps10_000_mirror.bvh',
    './data/animations/WalkingUpSteps11_000_mirror.bvh',
    './data/animations/WalkingUpSteps12_000_mirror.bvh',

    './data/animations/NewCaptures01_000.bvh',
    './data/animations/NewCaptures02_000.bvh',
    './data/animations/NewCaptures03_000.bvh',
    './data/animations/NewCaptures03_001.bvh',
    './data/animations/NewCaptures03_002.bvh',
    './data/animations/NewCaptures04_000.bvh',
    './data/animations/NewCaptures05_000.bvh',
    './data/animations/NewCaptures07_000.bvh',
    './data/animations/NewCaptures08_000.bvh',
    './data/animations/NewCaptures09_000.bvh',
    './data/animations/NewCaptures10_000.bvh',
    './data/animations/NewCaptures11_000.bvh',

    './data/animations/NewCaptures01_000_mirror.bvh',
    './data/animations/NewCaptures02_000_mirror.bvh',
    './data/animations/NewCaptures03_000_mirror.bvh',
    './data/animations/NewCaptures03_001_mirror.bvh',
    './data/animations/NewCaptures03_002_mirror.bvh',
    './data/animations/NewCaptures04_000_mirror.bvh',
    './data/animations/NewCaptures05_000_mirror.bvh',
    './data/animations/NewCaptures07_000_mirror.bvh',
    './data/animations/NewCaptures08_000_mirror.bvh',
    './data/animations/NewCaptures09_000_mirror.bvh',
    './data/animations/NewCaptures10_000_mirror.bvh',
    './data/animations/NewCaptures11_000_mirror.bvh',
]

#data_terrain = ['./data/animations/LocomotionFlat01_000.bvh']

""" Load Terrain Patches """

patches_database = np.load('patches.npz')
patches = patches_database['X'].astype(np.float32)
patches_coord = patches_database['C'].astype(np.float32)

""" Processing Functions """

def process_data(anim, phase, gait, type='flat'):
    #此函数用来准备训练数据所需的X,Y,以及control paras P
    """ Do FK """
    global_xforms = Animation.transforms_global(anim)  #（4086，31，4，4）
    
    #（4086，31，3）=（4086，31，3）/（4086，31，1）  4086为当前clip中的frame，31为关节点数目，3为X,Y,Zposition
    #global_xforms[:,:,3:,3]应该是（4，4）变换矩阵中的归一化系数
    global_positions = global_xforms[:,:,:3,3] / global_xforms[:,:,3:,3]
    global_rotations = Quaternions.from_transforms(global_xforms)
    
    """ Extract Forward Direction """
    
    sdr_l, sdr_r, hip_l, hip_r = 18, 25, 2, 7
    across = (
        (global_positions[:,sdr_l] - global_positions[:,sdr_r]) +  #沿着两肩膀的方向
        (global_positions[:,hip_l] - global_positions[:,hip_r]))
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]  
    
    """ Smooth Forward Direction """
    
    #np.cross(across, np.array([[0,1,0]])为求取与across和y轴（upward）都垂直的方向，即charactor的forward方向
    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    root_rotation = Quaternions.between(forward, 
        np.array([[0,0,1]]).repeat(len(forward), axis=0))[:,np.newaxis] 
    
    """ Local Space """
    
    local_positions = global_positions.copy()  #（4086，31，3）
    local_positions[:,:,0] = local_positions[:,:,0] - local_positions[:,0:1,0]  #当前位置的global pos减去root节点的global pos
    local_positions[:,:,2] = local_positions[:,:,2] - local_positions[:,0:1,2]
    
    local_positions = root_rotation[:-1] * local_positions[:-1]
    local_velocities = root_rotation[:-1] *  (global_positions[1:] - global_positions[:-1])
    local_rotations = abs((root_rotation[:-1] * global_rotations[:-1])).log()
    
    root_velocity = root_rotation[:-1] * (global_positions[1:,0:1] - global_positions[:-1,0:1])
    root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps
    
    """ Foot Contacts """
    
    fid_l, fid_r = np.array([4,5]), np.array([9,10])
    velfactor = np.array([0.02, 0.02])
    
    feet_l_x = (global_positions[1:,fid_l,0] - global_positions[:-1,fid_l,0])**2
    feet_l_y = (global_positions[1:,fid_l,1] - global_positions[:-1,fid_l,1])**2
    feet_l_z = (global_positions[1:,fid_l,2] - global_positions[:-1,fid_l,2])**2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)).astype(np.float)
    
    feet_r_x = (global_positions[1:,fid_r,0] - global_positions[:-1,fid_r,0])**2
    feet_r_y = (global_positions[1:,fid_r,1] - global_positions[:-1,fid_r,1])**2
    feet_r_z = (global_positions[1:,fid_r,2] - global_positions[:-1,fid_r,2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
    
    """ Phase """
    
    #相邻帧之间phase的变化量
    dphase = phase[1:] - phase[:-1]  
    #将负的phase变化量变成正的，如-0.1会变成1-0.1 = 0.9
    dphase[dphase < 0] = (1.0-phase[:-1]+phase[1:])[dphase < 0]
    
    """ Adjust Crouching Gait Value """
    
    if type == 'flat':
        crouch_low, crouch_high = 80, 130
        head = 16
        #gait shape：(4086,6)  head的y坐标（即upward）超过80太多，就clip为1，gait value就为0；反之为1 
        gait[:-1,3] = 1 - np.clip((global_positions[:-1,head,1] - 80) / (130 - 80), 0, 1)
        gait[-1,3] = gait[-2,3]   #最后一帧的gait value等于前一帧的gait value

    """ Start Windows """
    
    Pc, Xc, Yc = [], [], []
    
    for i in range(window, len(anim)-window-1, 1):
        
        #分别为root positions，directions 和gaits
        #当i = 60时，global_positions[i-window:i+window:10,0]即为global_positions[60+60:60+60:10,0]共12个sample
        #global_positions[i-window:i+window:10,0]：shape（12，3），12个sample的position
        #global_positions[i:i+1,0]：shape（1，3）第i个sample的position
        #root_rotation[i:i+1,0]：第i个sample中root相对于世界坐标系所做的rotation（12，1），每个rotation都用四元数表达
        #为什么？？
        #四元数左乘向量，代表对向量进行旋转？向量本身是global cord中的向量，左乘了rotation后，变成了相对于root cord的local向量？
        rootposs = root_rotation[i:i+1,0] * (global_positions[i-window:i+window:10,0] - global_positions[i:i+1,0])
        
        #forward本身也是global cord中的方向，左乘四元数后变成了相对于root坐标系的方向
        rootdirs = root_rotation[i:i+1,0] * forward[i-window:i+window:10]    
        rootgait = gait[i-window:i+window:10]
        
        Pc.append(phase[i])
        
        Xc.append(np.hstack([
                rootposs[:,0].ravel(), rootposs[:,2].ravel(), # Trajectory Pos 只需要x与z值，不需要y
                rootdirs[:,0].ravel(), rootdirs[:,2].ravel(), # Trajectory Dir
                rootgait[:,0].ravel(), rootgait[:,1].ravel(), # Trajectory Gait
                rootgait[:,2].ravel(), rootgait[:,3].ravel(), 
                rootgait[:,4].ravel(), rootgait[:,5].ravel(), 
                local_positions[i-1].ravel(),  # Joint Pos
                local_velocities[i-1].ravel(), # Joint Vel
                ]))
        
        rootposs_next = root_rotation[i+1:i+2,0] * (global_positions[i+1:i+window+1:10,0] - global_positions[i+1:i+2,0])
        rootdirs_next = root_rotation[i+1:i+2,0] * forward[i+1:i+window+1:10]   
        
        Yc.append(np.hstack([
                root_velocity[i,0,0].ravel(), # Root Vel X
                root_velocity[i,0,2].ravel(), # Root Vel Z
                root_rvelocity[i].ravel(),    # Root Rotation Vel
                dphase[i],                    # Change in Phase
                np.concatenate([feet_l[i], feet_r[i]], axis=-1), # Contacts
                rootposs_next[:,0].ravel(), rootposs_next[:,2].ravel(), # Next Trajectory Pos
                rootdirs_next[:,0].ravel(), rootdirs_next[:,2].ravel(), # Next Trajectory Dir
                local_positions[i].ravel(),  # Joint Pos
                local_velocities[i].ravel(), # Joint Vel
                local_rotations[i].ravel()   # Joint Rotation
                ]))
                                                
    return np.array(Pc), np.array(Xc), np.array(Yc)
    

""" Sampling Patch Heightmap """    

def patchfunc(P, Xp, hscale=3.937007874, vscale=3.0):
    '''
    应该是将patches组成的矩阵P移到以Xp为center的位置，并计算在此位置上的地形高度
    P:patches  shape:(10,128,128)
    Xp:feet_down_xz - feet_down_xz_mean  shape(201,2)
    return:terrain_down_y
    '''
    #将Xp移到patch的中心
    Xp = Xp / hscale + np.array([P.shape[1]//2, P.shape[2]//2])
    
    A = np.fmod(Xp, 1.0)   #将Xp中的元素分别对1.0求余:只保留小数部分
    ##对Xp中元素向下取整，并clip为（127，127）之间
    X0 = np.clip(np.floor(Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))  #shape（201，2）
    X1 = np.clip(np.ceil (Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))
    
    H0 = P[:,X0[:,0],X0[:,1]] #（10，201）
    H1 = P[:,X0[:,0],X1[:,1]]
    H2 = P[:,X1[:,0],X0[:,1]]
    H3 = P[:,X1[:,0],X1[:,1]]
    
    HL = (1-A[:,0]) * H0 + (A[:,0]) * H2
    HR = (1-A[:,0]) * H1 + (A[:,0]) * H3
    
    return (vscale * ((1-A[:,1]) * HL + (A[:,1]) * HR))[...,np.newaxis]
    

def process_heights(anim, nsamples=10, type='flat'):
    
    """ Do FK """
    
    global_xforms = Animation.transforms_global(anim)
    global_positions = global_xforms[:,:,:3,3] / global_xforms[:,:,3:,3]
    global_rotations = Quaternions.from_transforms(global_xforms)
    
    """ Extract Forward Direction """
    
    sdr_l, sdr_r, hip_l, hip_r = 18, 25, 2, 7
    across = (
        (global_positions[:,sdr_l] - global_positions[:,sdr_r]) + 
        (global_positions[:,hip_l] - global_positions[:,hip_r]))
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    """ Smooth Forward Direction """
    
    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    #[0,0,1]是z方即世界坐标系中的forward方向，
    #计算得到character的forward方向与世界坐标系forward方向之间的旋转
    root_rotation = Quaternions.between(forward, 
        np.array([[0,0,1]]).repeat(len(forward), axis=0))[:,np.newaxis]   

    """ Foot Contacts """
    
    fid_l, fid_r = np.array([4,5]), np.array([9,10])
    velfactor = np.array([0.02, 0.02])
    
    #global_positions[1:,fid_l,0]：shape:(4086,31,3)第一个维度上只选取index为1以后的，
    #第二个维度上只选取index为fid_l中数字（也就是4，5）的，第三个维度上只选取index为0的,得到的shape:(4085,2)
    #global_positions[1:...]-global_positions[:-1,...]的原因是：
    #[a2,a3,a4]-[a1,a2,a3] = [a2-a1,a3-a2,a4-a3]即为相邻两个frame之间，左脚在x方向上移动的距离。
    feet_l_x = (global_positions[1:,fid_l,0] - global_positions[:-1,fid_l,0])**2  
    feet_l_y = (global_positions[1:,fid_l,1] - global_positions[:-1,fid_l,1])**2
    feet_l_z = (global_positions[1:,fid_l,2] - global_positions[:-1,fid_l,2])**2
    #速度小于velfactor的视为foot contact，但是为什么要把三个相加？
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor))  #shape:4085,2
    
    feet_r_x = (global_positions[1:,fid_r,0] - global_positions[:-1,fid_r,0])**2
    feet_r_y = (global_positions[1:,fid_r,1] - global_positions[:-1,fid_r,1])**2
    feet_r_z = (global_positions[1:,fid_r,2] - global_positions[:-1,fid_r,2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor))
    
    feet_l = np.concatenate([feet_l, feet_l[-1:]], axis=0)
    feet_r = np.concatenate([feet_r, feet_r[-1:]], axis=0)
    
    """ Toe and Heel Heights """
    
    toe_h, heel_h = 4.0, 5.0
    
    """ Foot Down Positions """
    
    feet_down = np.concatenate([
        global_positions[feet_l[:,0],fid_l[0]] - np.array([0, heel_h, 0]),   #左脚跟触地时,减去heel_h后，得到触地position[x,0,z]
        global_positions[feet_l[:,1],fid_l[1]] - np.array([0,  toe_h, 0]),   #左脚趾触地
        global_positions[feet_r[:,0],fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[feet_r[:,1],fid_r[1]] - np.array([0,  toe_h, 0])
    ], axis=0)
    
    """ Foot Up Positions """
    
    feet_up = np.concatenate([
        global_positions[~feet_l[:,0],fid_l[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_l[:,1],fid_l[1]] - np.array([0,  toe_h, 0]),
        global_positions[~feet_r[:,0],fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_r[:,1],fid_r[1]] - np.array([0,  toe_h, 0])
    ], axis=0)
    
    """ Down Locations """
    
    feet_down_xz = np.concatenate([feet_down[:,0:1], feet_down[:,2:3]], axis=-1)  #将触地position[x,0,z]进行slice，变成[x,z]
    feet_down_xz_mean = feet_down_xz.mean(axis=0)
    feet_down_y = feet_down[:,1:2]
    feet_down_y_mean = feet_down_y.mean(axis=0)
    feet_down_y_std  = feet_down_y.std(axis=0)
        
    """ Up Locations """
        
    feet_up_xz = np.concatenate([feet_up[:,0:1], feet_up[:,2:3]], axis=-1)
    feet_up_y = feet_up[:,1:2]
    
    if len(feet_down_xz) == 0:
    
        """ No Contacts """ #说明当前motion cycle中没有与地面接触
    
        terr_func = lambda Xp: np.zeros_like(Xp)[:,:1][np.newaxis].repeat(nsamples, axis=0)
        
    elif type == 'flat':
        
        """ Flat """
        
        #Xp shape:(201,2)  np.zeros_like(Xp)[:,:1]的目的只是创建一个(201,1)的矩阵，最后得到的结果为（nsamples，201，1）
        #其中每个元素的值都等于feet_down_y_mean，也就是flat的地形，此时每个foot contact下地形height都为feet_down_y_mean
        terr_func = lambda Xp: np.zeros_like(Xp)[:,:1][np.newaxis].repeat(nsamples, axis=0) + feet_down_y_mean
    
    else:
        
        """ Terrain Heights """
        
        terr_down_y = patchfunc(patches, feet_down_xz - feet_down_xz_mean)
        terr_down_y_mean = terr_down_y.mean(axis=1)
        terr_down_y_std  = terr_down_y.std(axis=1)
        terr_up_y = patchfunc(patches, feet_up_xz - feet_down_xz_mean)
        
        """ Fitting Error """
        #最小化（去均值后的terr_down_y - 去均值后的feet_down_y）^2
        terr_down_err = 0.1 * ((
            (terr_down_y - terr_down_y_mean[:,np.newaxis]) -
            (feet_down_y - feet_down_y_mean)[np.newaxis])**2)[...,0].mean(axis=1)
        
        terr_up_err = (np.maximum(
            (terr_up_y - terr_down_y_mean[:,np.newaxis]) -
            (feet_up_y - feet_down_y_mean)[np.newaxis], 0.0)**2)[...,0].mean(axis=1)
        
        """ Jumping Error """
        
        if type == 'jumpy':
            terr_over_minh = 5.0
            terr_over_err = (np.maximum(
                ((feet_up_y - feet_down_y_mean)[np.newaxis] - terr_over_minh) -
                (terr_up_y - terr_down_y_mean[:,np.newaxis]), 0.0)**2)[...,0].mean(axis=1)
        else:
            terr_over_err = 0.0
        
        """ Fitting Terrain to Walking on Beam """
        
        if type == 'beam':

            beam_samples = 1
            beam_min_height = 40.0

            beam_c = global_positions[:,0]
            beam_c_xz = np.concatenate([beam_c[:,0:1], beam_c[:,2:3]], axis=-1)
            beam_c_y = patchfunc(patches, beam_c_xz - feet_down_xz_mean)

            beam_o = (
                beam_c.repeat(beam_samples, axis=0) + np.array([50, 0, 50]) * 
                rng.normal(size=(len(beam_c)*beam_samples, 3)))

            beam_o_xz = np.concatenate([beam_o[:,0:1], beam_o[:,2:3]], axis=-1)
            beam_o_y = patchfunc(patches, beam_o_xz - feet_down_xz_mean)

            beam_pdist = np.sqrt(((beam_o[:,np.newaxis] - beam_c[np.newaxis,:])**2).sum(axis=-1))
            beam_far = (beam_pdist > 15).all(axis=1)

            terr_beam_err = (np.maximum(beam_o_y[:,beam_far] - 
                (beam_c_y.repeat(beam_samples, axis=1)[:,beam_far] - 
                 beam_min_height), 0.0)**2)[...,0].mean(axis=1)

        else:
            terr_beam_err = 0.0
        
        """ Final Fitting Error """
        
        terr = terr_down_err + terr_up_err + terr_over_err + terr_beam_err
        
        """ Best Fitting Terrains """
        
        terr_ids = np.argsort(terr)[:nsamples]
        terr_patches = patches[terr_ids]
        terr_basic_func = lambda Xp: (
            (patchfunc(terr_patches, Xp - feet_down_xz_mean) - 
            terr_down_y_mean[terr_ids][:,np.newaxis]) + feet_down_y_mean)
        
        """ Terrain Fit Editing """
        
        terr_residuals = feet_down_y - terr_basic_func(feet_down_xz)
        terr_fine_func = [RBF(smooth=0.1, function='linear') for _ in range(nsamples)]
        for i in range(nsamples): terr_fine_func[i].fit(feet_down_xz, terr_residuals[i])
        #最终的terr_func应该是先进行basic func，后针对每个patch进行fine func
        terr_func = lambda Xp: (terr_basic_func(Xp) + np.array([ff(Xp) for ff in terr_fine_func]))
        
        
    """ Get Trajectory Terrain Heights """
    
    #root的global position
    root_offsets_c = global_positions[:,0]  #（201，3）
    #root的rotation（四元数）左乘right方向上长度25的方向就可得到：相对于root坐标系的向右25长度的方向。再加上root坐标系的global position
    #就得到右边25长度的global position
    root_offsets_r = (-root_rotation[:,0] * np.array([[+25, 0, 0]])) + root_offsets_c
    root_offsets_l = (-root_rotation[:,0] * np.array([[-25, 0, 0]])) + root_offsets_c
    
    #root_offsets_c[:,np.array([0,2])] shape:(201,2)
    root_heights_c = terr_func(root_offsets_c[:,np.array([0,2])])[...,0]  #(10,201)
    root_heights_r = terr_func(root_offsets_r[:,np.array([0,2])])[...,0]
    root_heights_l = terr_func(root_offsets_l[:,np.array([0,2])])[...,0]
    
    """ Find Trajectory Heights at each Window """
    
    root_terrains = []
    root_averages = []
    for i in range(window, len(anim)-window, 1):   #对每个window
        root_terrains.append(
            np.concatenate([
                root_heights_r[:,i-window:i+window:10],   #shape(10, 12)
                root_heights_c[:,i-window:i+window:10],   #shape(10, 12)
                root_heights_l[:,i-window:i+window:10]], axis=1))
        root_averages.append(root_heights_c[:,i-window:i+window:10].mean(axis=1))  #计算此window中的12个sample的root heights的均值
     
    root_terrains = np.swapaxes(np.array(root_terrains), 0, 1)   #(10,81,36)  81为window个数，每个window都有36个heights
    root_averages = np.swapaxes(np.array(root_averages), 0, 1)   #（10，81）
    
    return root_terrains, root_averages

""" Phases, Inputs, Outputs """
    
P, X, Y = [], [], []   #用于存储所有phase，输入，输出
            
for data in data_terrain:
    
    print('Processing Clip %s' % data)
    
    """ Data Types """
    
    if   'LocomotionFlat12_000' in data: type = 'jumpy'
    elif 'NewCaptures01_000'    in data: type = 'flat'
    elif 'NewCaptures02_000'    in data: type = 'flat'
    elif 'NewCaptures03_000'    in data: type = 'jumpy'
    elif 'NewCaptures03_001'    in data: type = 'jumpy'
    elif 'NewCaptures03_002'    in data: type = 'jumpy'
    elif 'NewCaptures04_000'    in data: type = 'jumpy'
    elif 'WalkingUpSteps06_000' in data: type = 'beam'
    elif 'WalkingUpSteps09_000' in data: type = 'flat'
    elif 'WalkingUpSteps10_000' in data: type = 'flat'
    elif 'WalkingUpSteps11_000' in data: type = 'flat'
    elif 'Flat' in data: type = 'flat'
    else: type = 'rocky'
    
    """ Load Data """
    
    anim, names, _ = BVH.load(data)
    anim.offsets *= to_meters
    anim.positions *= to_meters
    anim = anim[::2]

    """ Load Phase / Gait """
    
    phase = np.loadtxt(data.replace('.bvh', '.phase'))[::2]
    gait = np.loadtxt(data.replace('.bvh', '.gait'))[::2]

    """ Merge Jog / Run and Crouch / Crawl """
    
    gait = np.concatenate([
        gait[:,0:1],
        gait[:,1:2],
        gait[:,2:3] + gait[:,3:4],
        gait[:,4:5] + gait[:,6:7],
        gait[:,5:6],
        gait[:,7:8]
    ], axis=-1)

    """ Preprocess Data """
    #shape:(3965,)  (3965,311)  (3965,306)
    Pc, Xc, Yc = process_data(anim, phase, gait, type=type)  

    with open(data.replace('.bvh', '_footsteps.txt'), 'r') as f:
        footsteps = f.readlines()
    
    """ For each Locomotion Cycle fit Terrains """
    
    for li in range(len(footsteps)-1):
        #for each cycle
    
        curr, next = footsteps[li+0].split(' '), footsteps[li+1].split(' ')
        
        """ Ignore Cycles marked with '*' or not in range """
        
        if len(curr) == 3 and curr[2].strip().endswith('*'): continue
        if len(next) == 3 and next[2].strip().endswith('*'): continue
        if len(next) <  2: continue
        if int(curr[0])//2-window < 0: continue
        if int(next[0])//2-window >= len(Xc): continue 
        
        """ Fit Heightmaps """
        
        slc = slice(int(curr[0])//2-window, int(next[0])//2-window+1)
        
        #|--【左脚帧-----------|--右脚帧】--|     这样保证左脚和右脚帧都在采样区间内，且每个帧都要在前后各自有past/future samples  
        H, Hmean = process_heights(anim[
            int(curr[0])//2-window:
            int(next[0])//2+window+1], type=type)  #分别为（10，81，36） 以及（10，81）81为当前cycle中window数目

        for h, hmean in zip(H, Hmean):    
            Xh, Yh = Xc[slc].copy(), Yc[slc].copy()
            
            """ Reduce Heights in Input/Output to Match"""   
            #由于Xh shape 为（81，306），表示81个window，每个window中采集了306条数据，
            #其中，local_positions的index为120，121（y轴），122，因此Xh[:,121:213:3]表示所有window中的local_position的y值
            
            xo_s, xo_e = ((window*2)//10)*10+1, ((window*2)//10)*10+njoints*3+1  #（121～213）
            yo_s, yo_e = 8+(window//10)*4+1, 8+(window//10)*4+njoints*3+1  #（33～125）
            #给Xh中每个local position都减去当前window的平均高度，即去均值操作
            Xh[:,xo_s:xo_e:3] -= hmean[...,np.newaxis]
            Yh[:,yo_s:yo_e:3] -= hmean[...,np.newaxis]
            
            #h shape:(81,36)  h_mean shape:(81,)
            Xh = np.concatenate([Xh, h - hmean[...,np.newaxis]], axis=-1)  #由(81,306)变成(81,342)，加上了去均值后的terrian heights
            
            """ Append to Data """
            
            P.append(np.hstack([0.0, Pc[slc][1:-1], 1.0]).astype(np.float32))
            X.append(Xh.astype(np.float32))
            Y.append(Yh.astype(np.float32))
  
""" Clip Statistics """
  
print('Total Clips: %i' % len(X))
print('Shortest Clip: %i' % min(map(len,X)))
print('Longest Clip: %i' % max(map(len,X)))
print('Average Clip: %i' % np.mean(list(map(len,X))))

""" Merge Clips """

print('Merging Clips...')

Xun = np.concatenate(X, axis=0)
Yun = np.concatenate(Y, axis=0)
Pun = np.concatenate(P, axis=0)

print(Xun.shape, Yun.shape, Pun.shape)

print('Saving Database...')

np.savez_compressed('database.npz', Xun=Xun, Yun=Yun, Pun=Pun)

