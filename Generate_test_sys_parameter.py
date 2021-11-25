'''
author == Wenchen Chen
encoding  = utf-8
'''
import numpy as np
import pandas as pd
def generarate_test_sys_parameters(XFOV,YFOV,EPD,total_sys_num,yde_s2=0,zde_s2=0,yde_s4=0,zde_s4=0,yde_si=0,zde_si=0):
    '''
    分别生成系统参数和结构参数
    :param half_xfov_min:
    :param half_xfov_max:
    :param half_yfov_min:
    :param half_yfov_max:
    :param EPD_min:
    :param EPD_max:
    :param total_sys_num:
    :return:
    '''
    yde_s2_min = 50
    yde_s2_max = 100
    zde_s2_min = 150
    zde_s2_max = 250
    yde_s4_min = -50
    yde_s4_max = 0
    zde_s4_min = 150
    zde_s4_max = 250
    yde_si_min = -60
    yde_si_max = -16
    zde_si_min = -30
    zde_si_max = 30
    sys_parameters = np.zeros([total_sys_num, 9])

    if yde_s2 != 0:
        yde_s2_max = yde_s2
        yde_s2_min = yde_s2
    if zde_s2 != 0:
        zde_s2_max = zde_s2
        zde_s2_min = zde_s2
    if yde_s4 != 0:
        yde_s4_max = yde_s4
        yde_s4_min = yde_s4
    if zde_s4 != 0:
        zde_s4_max = zde_s4
        zde_s4_min = zde_s4
    if yde_si != 0:
        yde_si_max = yde_si
        yde_si_min = yde_si
    if zde_si != 0:
        zde_si_max = zde_si
        zde_si_min = zde_si


    for i in range(total_sys_num):
        sys_parameters[i, 0] = XFOV
        sys_parameters[i, 1] = YFOV
        sys_parameters[i, 2] = EPD
        sys_parameters[i, 3] = np.round(yde_s2_min + (yde_s2_max - yde_s2_min) * np.random.rand(1), 20)
        sys_parameters[i, 4] = np.round(zde_s2_min + (zde_s2_max - zde_s2_min) * np.random.rand(1), 20)
        sys_parameters[i, 5] = np.round(yde_s4_min + (yde_s4_max - yde_s4_min) * np.random.rand(1), 20)
        sys_parameters[i, 6] = np.round(zde_s4_min + (zde_s4_max - zde_s4_min) * np.random.rand(1), 20)
        sys_parameters[i, 7] = np.round(yde_si_min + (yde_si_max - yde_si_min) * np.random.rand(1), 20)
        sys_parameters[i, 8] = np.round(zde_si_min + (zde_si_max - zde_si_min) * np.random.rand(1), 20)
        print(sys_parameters[i, :])

# generarate_test_sys_parameters(XFOV=5,YFOV=5,EPD=20,total_sys_num=10,yde_s2=10)
#     保存预测出的系统参数



