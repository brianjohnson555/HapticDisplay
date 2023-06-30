import time
import numpy as np
import matplotlib.pyplot as plt
import PyDAQmx as nidaq
import sys
import math

F_CLK = 100000.0 # DOI Clock frequency (Circuit control sampling frequency, Hz)
channel_Num = 10 # [C1 D1 G1]

def PWM(totalSamples, freq, duty):
    PWMout = np.empty((0,1), dtype=np.uint8)
    numSamples = int(F_CLK//freq)  # samples per cycle
    cycles = int(totalSamples//numSamples)
    PWMcycle = np.zeros((numSamples,1), dtype=np.uint8)
    numHigh = int(math.floor(duty*numSamples))

    if(numHigh < 0):
        numHigh = 0
    PWMcycle[:numHigh] = 1 # High 
    for cycle in range(cycles):
        PWMout = np.append(PWMout, PWMcycle, axis=0)

    return PWMout

c = 60
C1 = [1, 1, 1, 1, 0, 0, 0, 0]
C1 = C1*c+[0]
D1 = [0, 0, 0, 0, 1, 1, 1, 1]
D1 = D1*c+[1]
C2 = [1, 1, 1, 1, 0, 0, 0, 0]
C2 = C2*c+[0]
D2 = [0, 0, 0, 0, 1, 1, 1, 1]
D2 = D2*c+[1]
C3 = [0, 0, 0, 0, 0, 0, 0, 0]
C3 = C2*c+[0]
D3 = [0, 0, 0, 0, 0, 0, 0, 0]
D3 = D2*c+[1]
G1 = [0, 0, 0, 0, 0, 0, 0, 0]
G2 = [1, 1, 1, 1, 1, 1, 1, 1]
G3 = [0, 0, 0, 0, 0, 0, 0, 0]
G4 = [0, 0, 0, 0, 0, 0, 0, 0]
G1 = G1*c+[1]
G2 = G2*c+[1]
G3 = G3*c+[1]
G4 = G4*c+[1]

PWMfreq = 1000
DO_out = np.empty((0,channel_Num), dtype=np.uint8)

for block,value in enumerate(C1):
    numSamples = F_CLK*0.05 # seconds per block
    DO_out_block = np.array([PWM(numSamples, PWMfreq, C1[block]), 
                             PWM(numSamples, PWMfreq, D1[block]),
                             PWM(numSamples, PWMfreq, C2[block]), 
                             PWM(numSamples, PWMfreq, D2[block]), 
                             PWM(numSamples, PWMfreq, C3[block]), 
                             PWM(numSamples, PWMfreq, D3[block]), 
                             PWM(numSamples, PWMfreq, G1[block]),
                             PWM(numSamples, PWMfreq, G2[block]),
                             PWM(numSamples, PWMfreq, G3[block]),
                             PWM(numSamples, PWMfreq, G4[block])])
    
    DO_out_block = np.reshape(DO_out_block,(channel_Num,-1))
    DO_out_block = np.transpose(DO_out_block)
    DO_out = np.append(DO_out, DO_out_block, axis=0)
DO_out[-1,:] = 0 # set to zero at end
print(DO_out)

DOoutLen = DO_out.shape[0]
print("DIOout Shape: ", DO_out.shape)
measureTime = (DOoutLen/F_CLK)
print("Total time = %.3f sec" % measureTime)
'''-------------------------------------------------------------------------------'''
''' Debug Tool'''
dispOutput = 1
if dispOutput:
    fig1 = plt.figure(figsize = (16,6))
    fig1.suptitle(("CLK Freq = %.0f Hz" % F_CLK), fontsize=12)
    ax = fig1.add_subplot(111)
    t = np.arange(DOoutLen)/F_CLK
    for i in range(channel_Num):
        ax.plot(t, DO_out[:,i]*0.5+i, '-')
        # ax.plot(t, DIOout[:, i] + i * 2, '.-')
    plt.show()
'''-------------------------------------------------------------------------------'''
if __name__ == '__main__':
    with nidaq.Task() as task1:
        # DAQ configuration
        task1.CreateDOChan("Dev1/port0/line0:5,Dev1/port0/line10:13", None, nidaq.DAQmx_Val_ChanPerLine)
        task1.CfgSampClkTiming(source="OnboardClock", rate=F_CLK, activeEdge=nidaq.DAQmx_Val_Rising,
                               sampleMode=nidaq.DAQmx_Val_FiniteSamps, sampsPerChan=DOoutLen)

        task1.WriteDigitalLines(numSampsPerChan=DOoutLen, autoStart=False,
                                timeout=nidaq.DAQmx_Val_WaitInfinitely, dataLayout=nidaq.DAQmx_Val_GroupByScanNumber,
                                writeArray=DO_out, reserved=None, sampsPerChanWritten=None)

        # ------------ start ------------ #
        task1.StartTask()
        print("Start sampling...")

        time.sleep(measureTime + 0.1)

        task1.StopTask()
        print("Task completed!")
        task1.ClearTask()

'''-------------------------------------------------------------------------------'''