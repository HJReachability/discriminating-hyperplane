import os, time
import math
import numpy as np
import scipy.io as sio

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data

from cs285_proj.dynamics.rabbit.rabbit_dynamics import RabbitDynamics
from cs285_proj.controllers.iocontroller import IOController
from cs285_proj.utils.utilFcns import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
import pdb
import copy
import tensorflow as tf
import imageio
import random

pi = np.pi

class RabbitEnv(gym.Env):
    def __init__(self, render=True, useTerrain=False, **kwargs):


        print('____________________init________________________')

        # Action is the joint torques
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype='float32')

        low_obs = np.array([-1, 0, -pi, -pi, -pi, -pi, -pi, -2, -5, -10, -10, -10, -10, -10, 0.2])
        high_obs = np.array([3, 2, pi, pi, pi, pi, pi, 5, 5, 10, 10, 10, 10, 10, 0.6])
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype='float32')

        ## Rabbit dynamics folder
        self.dynamicsFolder = './dynamics/rabbit/lib/librabbitdynamicsrd2.so'
        self.rabbbitDynamics = RabbitDynamics(self.dynamicsFolder)

        if (render):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)  # non-graphical version

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
        p.setRealTimeSimulation(0, self.physicsClient)

        ## Const Props:
        self.__ACTUATED_JOINT_NAMES = ['q1_right', 'q2_right', 'q1_left', 'q2_left']
        self.__BASE_NAMES = ['x_to_world', 'z_to_x', 'torso_to_z']
        self.__FIXED_JOINTS = ['right_foot', 'left_foot']  # , 'camera']
        self.__NON_FIXED_JOINTS = self.__BASE_NAMES + self.__ACTUATED_JOINT_NAMES
        self.FALL_THRESHOLD = 0.4

        self.extractJointId = lambda jointSet: [self.jointNameToId[a_] for a_ in jointSet]
        self.TIME_STEP = 1. / 240.

        self.GAITS = [0.4]
        # self.GAITS = [0.2, 0.4, 0.6]
        self.PHASES = {0: 'right_stance', 2: 'left_stance'}

        self.__buildGaitLibrary()

        self.MAX_STEPS = 4
        self.MAX_TIME = 800
        self.hingeV = 1000
        self.hingeB = 100000
        self.end_step = 0
        self.u_last = np.zeros([4, 1])
        self.pen_diff = 0

        # self.Reward_failed = -1e4
        self.Reward_failed = -10
        self.failed = 0
        self.max_action = 20
        self.exp_name = ''
        # self.VEL_THRESHOLD = -200
        self.VEL_THRESHOLD = -0.5


        self.ld_list = np.array([0.36, 0.39])

        num_gaits = len(self.GAITS)

        init_gait = self.GAITS[np.random.randint(low=0, high=num_gaits)]

        self.q0 = self.GAIT_LIBRARY['right_stance', 0.4, 0.4, 'x0']
        self.dq0 = self.GAIT_LIBRARY['right_stance', 0.4, 0.4, 'dx0']

        # self.ld = init_gait
        self.ld = 0.36
        self.t = 0

    def step(self, action):

        q, dq = self.computeStateObservation()
        u, V, gb, y = self.controller.calcControlInputsRD2(0, q, dq, self.ld)
        if self.phase == 'right_stance':
            uRL = self.max_action * action.reshape((4, 1))
        else:
            act = self.max_action * action.reshape((4, 1))
            uRL = np.copy(act)
            uRL[0:2] = act[2:4]
            uRL[2:4] = act[0:2]
        # print(uRL)
        u += uRL

        p.resetDebugVisualizerCamera(1.5, 0, 0, [q[0], 0, 1])

        isDone = self.computeDone(self.t, q, dq)
        self.failed = self.computeFailed(self.t, q, dq)

        if self.failed:
            print('XXXXXXXXXXX  Robot has fallen down  XXXXXXXXXXXXXXXXX')

        p.setJointMotorControlArray(bodyUniqueId=self.mdlId,
                                    jointIndices=self.extractJointId(self.__ACTUATED_JOINT_NAMES),
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=u)
        p.stepSimulation()

        # time.sleep(self.TIME_STEP)
        c = self.detectSwingFootImpact()
        self.t += self.TIME_STEP
        self.t_total += self.TIME_STEP
        tauVar = self.controller.phaseVarFunction(0, q)
        self.end_step = 0
        if c and tauVar > 0.9:
            print('contact detected')
            self.end_step = 1
            self.numSteps += 1
            print('Step No = {}'.format(self.numSteps))

            self.t = 0
            # Update phase
            if self.phase == 'right_stance':
                self.phase = 'left_stance'
            else:
                self.phase = 'right_stance'
            
            print('desired step length = {}, actual step length = {}' .format(self.ld, self.computeStepLength()))
            self.ld = 0.35 + 0.05 * random.random()
            # self.ld = self.ld_list[self.numSteps%2]

            gaitParams = {}
            gaitParams['alpha2'] = self.GAIT_LIBRARY[self.phase, 0.4, 0.4, 'alpha']
            gaitParams['theta2'] = self.GAIT_LIBRARY[self.phase, 0.4, 0.4, 'theta']
            self.controller.setPhase(self.phase, gaitParams)
            # self.ld =  0.25 + 0.15*random.random()


            # print('t = {}'.format(self.t))


            # img = self.render()
            # imageio.imwrite('images/{}/im_{}.png'.format(self.exp_name, self.numSteps), img)

        # Compute Observation before going into next step
        obs = self.computeObservation()

        qnext, dqnext = self.computeStateObservation()
        _, Vnext, gbnext, _ = self.controller.calcControlInputsRD2(0, qnext, dqnext, self.ld)

        reward = self.computeReward(u, V, Vnext, gb, gbnext)

        # self.V_last = np.copy(self.V)

        # Step Length
        sL = self.computeStepLength()

        Vdot = (Vnext - V) / self.TIME_STEP
        Delta = Vdot + V * self.controller.clfqp.gamma / self.controller.clfqp.epsilon
        Bdot1 = (gbnext[0] - gb[0]) / self.TIME_STEP
        DeltaB1 = Bdot1 + gb[0] * self.controller.clfqp.gamma_CBF
        Bdot2 = (gbnext[1] - gb[1]) / self.TIME_STEP
        DeltaB2 = Bdot2 + gb[1] * self.controller.clfqp.gamma_CBF

        return obs, reward, isDone, {'input': u, 'output': y, 'V': V, 'B': gb, 'Vdot': Vdot, 'DeltaV': Delta,
                                     'DeltaB1': DeltaB1, 'DeltaB2': DeltaB2, 'sL': sL, 'ld': self.ld}

    def computeDone(self, t, q, dq):

        done1 = q[1] < self.FALL_THRESHOLD  #
        done2 = dq[0] < self.VEL_THRESHOLD
        done3 = t >= 2.5  # or t<=3*self.TIME_STEP
        done4 = self.numSteps >= self.MAX_STEPS
        done5 = self.t_total >= self.MAX_TIME
        # done6 = np.any(np.greater(np.absolute(dq[2:]),self.spd_thr))
        # If robot falls down:
        return done1 or done2 or done3 or done4 or done5  # or done6

    def computeFailed(self, t, q, dq):

        done1 = q[1] < self.FALL_THRESHOLD  #
        done2 = dq[0] < self.VEL_THRESHOLD
        done3 = t >= 2.5  # or t<=3*self.TIME_STEP
        # done4 = np.any(np.greater(np.absolute(dq[2:]),self.spd_thr))
        # If robot falls down:
        return done1 or done2 or done3  # or done4

    def computeStateObservation(self):

        ## This gets the positions and velocities of all the joints
        x = np.asarray(p.getJointStates(self.mdlId, self.extractJointId(self.__NON_FIXED_JOINTS)), dtype=object)
        q = np.reshape(x[:, 0], (7, 1)).astype('double')
        dq = np.reshape(x[:, 1], (7, 1)).astype('double')
        return q, dq

    def computeObservation(self):

        if self.phase == 'right_stance':
            q, dq = self.computeStateObservation()
            obs = np.concatenate((q, dq, np.array([np.array([self.ld])])), axis=0)
        else:
            q, dq = self.computeStateObservation()
            new_q = np.copy(q)
            new_dq = np.copy(dq)
            new_q[3:5] = q[5:7]
            new_q[5:7] = q[3:5]
            new_dq[3:5] = dq[5:7]
            new_dq[5:7] = dq[3:5]
            obs = np.concatenate((new_q, new_dq, np.array([np.array([self.ld])])), axis=0)
            '''act = self.max_action*action.reshape((4,1))
            #print(act)
            #print(act[2:4])
            uRL = np.copy(act)
            uRL[0:2] = act[2:4]
            uRL[2:4] = act[0:2]'''
        return obs.squeeze()

    def computeReward(self, u, V, Vnext, gb, gbnext):

        Vdot = (Vnext - V) / self.TIME_STEP
        DeltaV = Vdot + V * self.controller.clfqp.gamma / self.controller.clfqp.epsilon

        Bdot0 = (gbnext[0] - gb[0]) / self.TIME_STEP
        DeltaB0 = Bdot0 + gb[0] * self.controller.clfqp.gamma_CBF

        Bdot1 = (gbnext[1] - gb[1]) / self.TIME_STEP
        DeltaB1 = Bdot1 + gb[1] * self.controller.clfqp.gamma_CBF

        loss = -np.min(np.array([0, self.hingeB * DeltaB1])) - np.min(np.array([0, self.hingeB * DeltaB0])) + np.max(
            np.array([0, self.hingeV * DeltaV[0, 0]])) + np.linalg.norm(u) ** 2
        loss += self.pen_diff * np.linalg.norm(u - self.u_last) ** 2
        self.u_last = np.copy(u)
        reward = -loss  # + (~failed)*self.Reward_not_failed
        # reward = np.clip(reward,a_min=-1e6,a_max=0)
        # # reward = -np.log10(-reward)
        # reward = reward/1e6
        reward = np.clip(reward, a_min=-5e6, a_max=0)
        reward = reward / 5e6
        reward = reward + self.failed * self.Reward_failed
        reward = reward * (self.end_step < 1)
        reward += 1
        # print('reward', reward)
        # if DeltaB1 <0:
        #     pdb.set_trace()

        return reward

    ### Reset:
    def reset(self, true_dyn=False):
        # reset is called once at initialization of simulation
        print('-------------------Reset------------------------')

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)  # m/s^2
        p.setPhysicsEngineParameter(erp=1e-6, contactERP=1e-6, frictionERP=1e-6, fixedTimeStep=1. / 240)
        p.resetDebugVisualizerCamera(5.5, 0, 0, [3.5, 0, 1])

        self.planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(self.planeId, -1, restitution=0.1, frictionAnchor=False,
                         lateralFriction=1)  # , contactStiffness=1e4, contactDamping=5e3)#, lateralFriction=0.9)

        self.planeId = np.array([[self.planeId]])
        # Initialize robot
        startPos = [0, 0, 0]
        startOrien = p.getQuaternionFromEuler([0, 0, 0])
        folderPath, _ = os.path.split(os.path.abspath(os.path.dirname(__file__)))
        # self.mdlId = p.loadURDF(os.path.join(folderPath, "data", "urdf", "five_link_walker_with_camera.urdf"),
        #                         startPos, startOrien, useFixedBase=False)
        self.mdlId = p.loadURDF(os.path.join(folderPath, "data", "urdf", "five_link_walker.urdf"),
                                        startPos, startOrien, useFixedBase =False)

        self.__createJointNameToId()

        # setting up physics
        p.setRealTimeSimulation(0, self.physicsClient)

        c1 = self.__detectRightFootContact()
        c2 = self.__detectLeftFootContact()

        # Initial phase of the robot
        self.phase = 'right_stance'

        # self.ld = 0.36

        num_gaits = len(self.GAITS)

        init_gait = self.GAITS[np.random.randint(low=0, high=num_gaits)]

        # self.q0 = self.GAIT_LIBRARY['right_stance', init_gait, init_gait, 'x0']
        # self.dq0 = self.GAIT_LIBRARY['right_stance', init_gait, init_gait, 'dx0']

        self.q0 = 0.2*self.GAIT_LIBRARY['right_stance', init_gait, init_gait, 'x0'] + 0.8*self.GAIT_LIBRARY['right_stance', 0.4, 0.4, 'x0']
        self.dq0 = 0.2*self.GAIT_LIBRARY['right_stance', init_gait, init_gait, 'dx0'] + 0.8*self.GAIT_LIBRARY['right_stance', 0.4, 0.4, 'dx0']
        # self.q0 += 2*0.01*(np.random.rand(7) - 0.5)*2
        # self.dq0 += 2*0.005*(np.random.rand(7) - 0.5)*2
        # self.dq0 += 2*0.005*(np.random.rand(7) - 0.3)*2
        # pdb.set_trace()

        qInit = self.q0
        dqInit = self.dq0

        for i, jId in enumerate(self.extractJointId(self.__NON_FIXED_JOINTS)):
            p.resetJointState(self.mdlId, jId, qInit[i], targetVelocity=0)

        while c1 == 0 and c2 == 0:
            for i, joint in enumerate(self.extractJointId(self.__ACTUATED_JOINT_NAMES)):
                p.setJointMotorControl2(self.mdlId, joint, p.PD_CONTROL, targetPosition=qInit[i + 3], targetVelocity=0,
                                        force=150)
            p.stepSimulation()
            c1 = self.__detectRightFootContact()
            c2 = self.__detectLeftFootContact()

        count = 0
        # ensure velocities are zero
        while count < 100:
            p.stepSimulation()
            count += 1

        q0 = []
        for i, jId in enumerate(self.extractJointId(self.__NON_FIXED_JOINTS)):
            q0.append(p.getJointState(self.mdlId, jId)[0])

        ## IO controller:

        self.numSteps = 0
        # self.ld = init_gait

        gaitParams = {}
        # gaitParams['alpha2'] = self.GAIT_LIBRARY['right_stance', init_gait, init_gait, 'alpha']
        # gaitParams['theta2'] = self.GAIT_LIBRARY['right_stance', init_gait, init_gait, 'theta']
        gaitParams['alpha2'] = self.GAIT_LIBRARY['right_stance', 0.4, 0.4, 'alpha']
        gaitParams['theta2'] = self.GAIT_LIBRARY['right_stance', 0.4, 0.4, 'theta']
        # self.controller = IOController(self.phase, gaitParams, k, self.dynamicsFolder)
        self.controller = IOController(self.phase, gaitParams, self.dynamicsFolder)

        if true_dyn:
            self.control_true_dyn()

        # Time step
        self.TIME_STEP = 1 / 1000.

        p.setPhysicsEngineParameter(erp=1e-6, contactERP=1e-6, frictionERP=1e-6,
                                    fixedTimeStep=self.TIME_STEP)  # , numSolverIterations=50, numSubSteps=10)

        ## Enables torque control mode
        self.__torqueControlMode()
        #
        for i, jId in enumerate(self.extractJointId(self.__BASE_NAMES)):
            p.resetJointState(self.mdlId, jId, q0[i], targetVelocity=dqInit[i])

        for i, jId in enumerate(self.extractJointId(self.__ACTUATED_JOINT_NAMES)):
            p.resetJointState(self.mdlId, jId, qInit[i + 3], targetVelocity=dqInit[i + 3])

        ## Compute initial CLF
        # u, V, _ = self.controller.calcControlInputsRD2(0,qInit.reshape((7,1)), dqInit.reshape((7,1)), 0.4)
        # self.V_last = np.copy(self.V)
        self.t_total = 0
        self.u_last = np.zeros([4, 1])

        # Return initial observation
        obs = self.computeObservation()

        self.t = 0

        return obs

    def render(self, mode='human', close=False):

        w, h, viewMat, projMat, _, _, _, _, _, _, _, _ = p.getDebugVisualizerCamera()
        _, _, img, _, _ = p.getCameraImage(w, h, viewMat, projMat)
        return img

    def simulate(self, numSteps=10, true_dyn=False):
        self.reset()
        if true_dyn == True:
            self.control_true_dyn()
        done = False
        while self.numSteps < numSteps and not done:
            action = np.zeros((4,))
            _, rew, done, _ = self.step(action)
            # print(rew)

    def control_true_dyn(self):
        gaitParams = {}
        gaitParams['alpha2'] = self.GAIT_LIBRARY['right_stance', 0.4, 0.4, 'alpha']
        gaitParams['theta2'] = self.GAIT_LIBRARY['right_stance', 0.4, 0.4, 'theta']
        self.dynamicsFolder = './dynamics/rabbit/lib/librabbitdynamicsrd2_scale2.so'
        self.controller = IOController(self.phase, gaitParams, self.dynamicsFolder)

    ## Some helper functions
    def getSwingLegPosition(self):

        if self.phase == 'right_stance':
            pSwing = p.getLinkState(self.mdlId, self.jointNameToId['left_foot'], computeForwardKinematics=True)[4]

        else:
            pSwing = p.getLinkState(self.mdlId, self.jointNameToId['right_foot'], computeForwardKinematics=True)[4]

        return pSwing

    def __createJointNameToId(self):
        jointNameToId = {}
        for i in range(p.getNumJoints(self.mdlId)):
            jointInfo = p.getJointInfo(self.mdlId, i)
            jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]

        self.jointNameToId = jointNameToId

    def computeStepLength(self):
        pR = p.getLinkState(self.mdlId, self.jointNameToId['right_foot'], computeForwardKinematics=True)[4]
        pL = p.getLinkState(self.mdlId, self.jointNameToId['left_foot'], computeForwardKinematics=True)[4]

        return np.abs(pR[0] - pL[0])

    def getStanceFootPosition(self):

        if self.phase == 'right_stance':
            pStance = p.getLinkState(self.mdlId, self.jointNameToId['right_foot'], computeForwardKinematics=True)[4]
        else:
            pStance = p.getLinkState(self.mdlId, self.jointNameToId['left_foot'], computeForwardKinematics=True)[4]

        return pStance

    def detectSwingFootImpact(self):
        isContact = 0
        if self.phase == 'right_stance':
            # Left foot contact
            for i in range(len(self.planeId)):
                c = p.getContactPoints(self.mdlId, self.planeId[i], self.jointNameToId[self.__FIXED_JOINTS[1]], -1)
                isContact += len(c)
        else:
            # Right Foot Contact
            for i in range(len(self.planeId)):
                c = p.getContactPoints(self.mdlId, self.planeId[i], self.jointNameToId[self.__FIXED_JOINTS[0]], -1)
                isContact += len(c)

        return isContact > 0

    def __detectRightFootContact(self):
        # print('Right Foot contact')
        isContact = 0
        for i in range(len(self.planeId)):
            c = p.getContactPoints(self.mdlId, self.planeId[i], self.jointNameToId[self.__FIXED_JOINTS[0]], -1)
            isContact += len(c)
        return isContact > 0

    def __detectLeftFootContact(self):
        # print('Left Foot contact')
        isContact = 0

        # Left foot contact
        for i in range(len(self.planeId)):
            c = p.getContactPoints(self.mdlId, self.planeId[i], self.jointNameToId[self.__FIXED_JOINTS[1]], -1)
            isContact += len(c)
        return isContact > 0

    ## might not need the this, but could be helpful for loading gaits...
    def __buildGaitLibrary(self, gaitFolder='./data/gaits/rabbit/walking_gaits_rd2/', saveGaitLib=False):
        print('Building Gait Library .....')
        self.GAIT_LIBRARY = {}
        for l0 in self.GAITS:
            for l1 in self.GAITS:
                for pid in self.PHASES:
                    l0_ = str(l0).replace('.', '_')
                    l1_ = str(l1).replace('.', '_')
                    # pdb.set_trace()
                    gaitFile = gaitFolder + 'optimizationResult_' + l0_ + '_' + l1_ + '.mat'
                    print('Adding gait: \n Phase = {}\n l0 = {}, l1 = {}\n'.format(self.PHASES[pid], str(l0), str(l1)))
                    print(
                        '---------------------------------------------------------------------------------------------')
                    mat_contents = sio.loadmat(gaitFile, squeeze_me=True)
                    if pid == 0:  # [Right Stance]
                        # In this case l0 = ldes
                        ldes = l0
                        lcur = l1

                    else:
                        ldes = l1
                        lcur = l0

                    self.GAIT_LIBRARY[self.PHASES[pid], lcur, ldes, 'alpha'] = readMatContents(mat_contents, 'params',
                                                                                               'atime', pid)
                    self.GAIT_LIBRARY[self.PHASES[pid], lcur, ldes, 'theta'] = readMatContents(mat_contents, 'params',
                                                                                               'ptime', pid)
                    self.GAIT_LIBRARY[self.PHASES[pid], lcur, ldes, 'x0'] = readMatContents(mat_contents, 'states', 'x',
                                                                                            pid)[:, 0]
                    self.GAIT_LIBRARY[self.PHASES[pid], lcur, ldes, 'dx0'] = readMatContents(mat_contents, 'states',
                                                                                             'dx', pid)[:, 0]
        return

    def __torqueControlMode(self):

        # Required for Torque control:
        jointFrictionForce = 0
        for joint in self.extractJointId(self.__ACTUATED_JOINT_NAMES):
            p.setJointMotorControl2(self.mdlId, joint, p.VELOCITY_CONTROL, force=jointFrictionForce)

        jointFrictionForce = 0
        for joint in self.extractJointId(self.__BASE_NAMES):
            p.setJointMotorControl2(self.mdlId, joint, p.VELOCITY_CONTROL, force=jointFrictionForce)
        return
