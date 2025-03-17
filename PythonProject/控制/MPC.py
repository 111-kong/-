########################################################################################################
#
#  This code provides users of Ansys Aqwa with examples of how to use the interface allowing them to
#  apply forces onto Aqwa structures at runtime. Please do not use if you are not a registered user of Aqwa.
#
########################################################################################################
from keras import Model
import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute
from lstm import sliding_window_lstm_predicts as sliding_window_lstm_predicts
from AqwaServerMgr import *
import json
import numpy as np
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from lstm import sliding_window_lstm_train  # 新增导入
########################################################################################################
#
# The user would typically define one or more functions. As an example, we define three of them below :
#
#      UF1, UF2, and UF3
#
#      After that, at the bottom of the file, we instanciate the Server
#
########################################################################################################
# Simple example
# Elastic force maintaining the structure around (0.0)
# AddMass is initialized and kept with a funny pattern in it's values (easily spottable in a output listing)
def UF1(Analysis, Mode, Stage, Time, TimeStep, Pos, Vel, nn_pred_pos=None, sliding_window_lstm_predicts=None, lstm=None):
    # Check that input file is the one expected without .DAT extension

    Error   = 0
    ExpectedFileName = "TIMERESPONSE"
    ActualFileName = Analysis.InputFileName.split("\\")[-1] # We strip the path part of the filename for easy comparison
    if (ActualFileName!=ExpectedFileName):
        print("Error. Incorrect input file !")
        print("Expected : "+ ExpectedFileName)
        print("Actual : "+ActualFileName)
        Error = 1    # Will cause Aqwa to stop

    # If this passed, we create an empty container for AddMass and Force
    AddMass = BlankAddedMass(Analysis.NOfStruct)
    Force   = BlankForce(Analysis.NOfStruct)

    # User defined code here
    """
    先加载lstm模型的预测结果
    再加载标准化器
    再加载MPC控制器
    最后调用MPC控制器的predict函数，传入当前位置，得到控制力
    """
    # 加载训练好的LSTM模型
    if not hasattr(UF1, 'lstm_model'):
        # 获取前1000秒的位移数据
        motion_time_series = Analysis.GetDisplacementHistory()[:1000]
        # 训练LSTM模型
        UF1.lstm_model, _, _, _, _, UF1.scaler = sliding_window_lstm_train(motion_time_series, window_size=50)
    # MPC控制逻辑
    class MPCController:
        def __init__(self, horizon=10, dt=0.1):
            self.horizon = horizon  # 预测时域
            self.dt = dt  # 时间步长
            self.u_min = -10000  # 控制力下限
            self.u_max = 10000  # 控制力上限
            self.Q = 1.0  # 状态误差权重
            self.R = 0.1  # 控制力变化权重
            self.prev_force = np.zeros(6)  # 上一次的控制力
            self.control_started = False  # 添加控制开始标志

            # 初始化LSTM预测器
            self.lstm_predictor = sliding_window_lstm_predicts

        def predict(self, current_pos, nn_pred_pos):
            # 如果控制还未开始，返回零控制力
            if not self.control_started:
                return np.zeros(6)

            # 对当前位置进行标准化
            scaled_pos = self.scaler.transform(current_pos.reshape(1, -1))
            # 使用LSTM进行预测
            nn_pred_pos = self.lstm_predictor.predict(scaled_pos)
            # 反标准化预测结果
            nn_pred_pos = self.scaler.inverse_transform(nn_pred_pos)

            # 定义优化问题
            from scipy.optimize import minimize
            nn_pred_pos = self.lstm_predictor(current_pos)  # 调用LSTM预测函数

            # 代价函数
            def cost_function(u):
                # 预测误差
                error = nn_pred_pos - (current_pos + u * self.dt)

                # 控制力变化
                delta_u = u - self.prev_force


                # 总代价
                cost = self.Q * np.sum(error ** 2) + self.R * np.sum(delta_u ** 2)
                return cost

            # 约束条件
            constraints = [{'type': 'ineq', 'fun': lambda u: self.u_max - u},
                           {'type': 'ineq', 'fun': lambda u: u - self.u_min}]

            # 初始猜测
            u0 = self.prev_force

            # 求解优化问题
            res = minimize(cost_function, u0, method='SLSQP', constraints=constraints)

            if res.success:
                control_force = res.x
                self.prev_force = control_force
            else:
                # 如果优化失败，使用上一次的控制力
                control_force = self.prev_force

            return control_force
    # Now return the results
    # 初始化MPC控制器
    mpc = MPCController()
 
    # 应用MPC控制
    for s in range(Analysis.NOfStruct):
        # 获取当前结构的位置
        current_pos = Pos[s]
        # 仅在1000秒后开始MPC控制 # 仅在1000秒后开始MPC控
        if Time >= 1000:
            # 获取MPC控制力
            mpc_force = mpc.predict(current_pos, nn_pred_pos[s])
            # 将控制力应用到结构上
            for dof in range(6):
                Force[s][dof] = mpc_force[dof]


    return Force,AddMass,Error

########################################################################################################
# Example applying a vertical Force away from the centre of gravity
#
def UF2(Analysis,Mode,Stage,Time,TimeStep,Pos,Vel):
    AddMass = BlankAddedMass(Analysis.NOfStruct)
    Force   = BlankForce(Analysis.NOfStruct)
    Error   = 0

    ExpectedFileName = "TIMERESPONSE"
    ActualFileName = Analysis.InputFileName.split("\\")[-1] # We strip the path part of the filename for easy comparison
    if (ActualFileName!=ExpectedFileName):
        print("Error. Incorrect input file !")
        print("Expected : "+ ExpectedFileName)
        print("Actual : "+ActualFileName)
        Error = 1    # Will cause Aqwa to stop


    # Note, we use the R_Control values specified in the DAT file if they are not zero.
    if (Analysis.R_Control[5]==0.0):
        coef = -10000.0
    else:
        coef = Analysis.R_Control[5] # Should be set to 30000 in the Aqwa input file.

    VertForce   = ( 0, 0, coef)

    # Application point coordinates in Deck 1's system of coordinates (using Node 3 of AD_PYTHONUSERFORCE.DAT)

    DefPos  = ( 0, 17.53, 0)

    CurPos = Analysis.GetNodeCurrentPosition(Struct=0,
                                             DefAxesX=DefPos[0],
                                             DefAxesY=DefPos[1],
                                             DefAxesZ=DefPos[2])

    ExtraForce = Analysis.ApplyForceOnStructureAtPoint(Struct=0,
                                                       FX=VertForce[0],
                                                       FY=VertForce[1],
                                                       FZ=VertForce[2],
                                                       AppPtX=CurPos[0],
                                                       AppPtY=CurPos[1],
                                                       AppPtZ=CurPos[2])

    # Add the extra force generated to initially empty force
    # (BlankForce and BlankAddedMass classes support algebraic operations (+, -, and scalar multiplication)

    Force += ExtraForce

    # Note : Looking at the .LIS file, the user should be able to verify that we have just set
    #        this force so that it exactly negates the mooring force associated to the FORC card in deck 14.
    # Now return the results

    return Force,AddMass,Error

########################################################################################################



########################################################################################################
#
#  Finally instanciating the server and asking it to run each of our functions.
#  This is going to start the server which is going to wait for an incoming connection
#  from the Aqwa program. It will then use the first function UF1 for that particular Aqwa run.
#  The process will then repeat for two more Aqwa runs, using UF2 and UF3.
#

Server = AqwaUserForceServer()

for UF in [UF1]:
    try:
        print("Now running user function {0}".format(UF.__name__))
        Server.Run(UF)
    except Exception as E: # If an error occurred, we print it but continue
        print("Caught error : ",E)
        print("Skipping to next case")
