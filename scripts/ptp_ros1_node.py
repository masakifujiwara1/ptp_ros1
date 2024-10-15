import os
import math
import sys
import torch
import pickle
import argparse
import torch.distributions.multivariate_normal as torchdist
import rospy
import numpy as np
from ptp_msgs.msg import PedestrianArray
from utils import *
from metrics import *
from model_depth_fc_fix import GAT_TimeSeriesLayer
import copy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from marker import *
import time

class PtpRos1Node:
    def __init__(self):
        rospy.init_node('ptp_ros1_node')
        self.ped_sub = rospy.Subscriber('ped_seq', PedestrianArray, self.ped_callback)
        self.data_array = None

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.path = '../checkpoint/social-stgcnn-eth'
        self.model_path = self.path + '/val_best.pth'
        self.obs_seq_len = 8
        self.pred_seq_len = 12

        rospy.loginfo('Model initialized')
        self.model = GAT_TimeSeriesLayer(in_features=2, hidden_features=16, out_features=5, obs_seq_len=self.obs_seq_len, pred_seq_len=self.pred_seq_len, num_heads=2).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        rospy.loginfo('Model loaded')

        self.debug_flag = False

        if self.debug_flag:
            with open('data_array.pkl','rb') as f: 
                self.data_array = pickle.load(f)
                rospy.loginfo(f'load array: {self.data_array}, shape: {self.data_array.shape}')

            with open('raw_data_dict.pkl','rb') as f: 
                self.raw_data_dict = pickle.load(f)
                rospy.loginfo(f'raw_data_dict: {self.raw_data_dict}')

        self.dset_test = TrajectoryDataset(
                obs_len=self.obs_seq_len,
                pred_len=self.pred_seq_len,
                skip=1, norm_lap_matr=True)

        self.marker_array_pub = rospy.Publisher('viz_marker', MarkerArray, queue_size=10)
        self.marker_array = MarkerArray()
        self.timer = rospy.Timer(rospy.Duration(0.4), self.prediction)

    def ped_callback(self, msg):
        self.data_array = np.array(msg.data, dtype=np.dtype(msg.dtype)).reshape(msg.shape)

    def prediction(self, event):
        self.marker_array = MarkerArray()
        raw_data_dict = self.test()
        if raw_data_dict == 0:
            return

        num_of_ped = raw_data_dict[0]['obs'].shape[1]
        obs_pred_seq = np.concatenate((raw_data_dict[0]['obs'], raw_data_dict[0]['pred']), axis=0)

        id_cnt = 0
        for i in range(num_of_ped):
            line_strip, points_marker = CreateMarker(obs_pred_seq[:, i, :], id_cnt)
            self.marker_array.markers.append(line_strip)
            self.marker_array.markers.append(points_marker)
            id_cnt += 2
        self.marker_array_pub.publish(self.marker_array)

    def test(self, KSTEPS=1):
        self.model.eval()
        ade_bigls = []
        fde_bigls = []
        raw_data_dict = {}
        step = 0 
        try:
            obs_traj, obs_traj_rel, non_linear_ped, loss_mask, V_obs, A_obs = self.dset_test.processed_data(self.data_array)
        except:
            return 0
        V_obs = V_obs.unsqueeze(0).to(self.device)
        A_obs = A_obs.unsqueeze(0).to(self.device)
        obs_traj = obs_traj.unsqueeze(0).to(self.device)
        obs_traj_rel = obs_traj_rel.unsqueeze(0).to(self.device)

        torch.cuda.synchronize()
        start = time.time()
        V_pred = self.model(V_obs, A_obs)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        print(f'elapsed_time: {elapsed_time}')

        V_pred = V_pred.squeeze(0)
        num_of_objs = obs_traj_rel.shape[1]
        V_pred =  V_pred[:,:num_of_objs,:]

        sx = torch.exp(V_pred[:,:,2]) 
        sy = torch.exp(V_pred[:,:,3]) 
        corr = torch.tanh(V_pred[:,:,4])

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).to(self.device)
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)

        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_obs = V_obs[:, :, :, :2]
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze(0).copy(), V_x[0,:,:].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        V_pred = mvnormal.sample()
        V_pred = V_pred.unsqueeze(0)

        V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze(0).copy(), V_x[-1,:,:].copy())
        raw_data_dict[step]['pred'] = copy.deepcopy(V_pred_rel_to_abs)

        return raw_data_dict

if __name__ == '__main__':
    try:
        rospy.init_node('ptp_ros1_node')
        node = PtpRos1Node()
        rospy.spin()
    except KeyboardInterrupt:
        print("ctrl-C")
    finally:
        rospy.signal_shutdown('Shutting down node')
