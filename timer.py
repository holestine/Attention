from contextlib import ContextDecorator 
import numpy as np 
import pandas as pd
import time 

class Timer(ContextDecorator): 
    """
    This class is used for timing various stages of execution in a Python script.
    """

    time_tracker = {}

    def __init__(self, phase = ""): 
        if phase != "":
            if not phase in self.time_tracker:  
                self.time_tracker[phase] = {'times':[]}  
            self.phase = phase  

    def __enter__(self):  
        self.start = time.time()  

    def __exit__(self, *args):  
        self.end = time.time()  
        self.time_tracker[self.phase]['times'].append(self.end - self.start)  

    def report_phase(self, phase, times): 
        print(f'{phase} ran {len(times)} times')  
        #print(f'\tTimes: {times}' ) 
        print(f'\tMin time was {1000 * np.min(times)} at index {times.index(np.min(times))}')  
        print(f'\tMax time was {1000 * np.max(times)} at index {times.index(np.max(times))}') 
        print(f'\tAverage time was {1000 * np.mean(times)}')  
        print(f'\tTotal time was {1000 * np.sum(times)} ms\n')

    def report_phases(self, filename=None):
        if filename:
            data = []

        for phase, times in self.time_tracker.items():  
            self.report_phase(phase, times['times'])
            if filename:
                data.append([phase, np.min(times['times']), np.max(times['times']), np.mean(times['times']), np.sum(times['times'])])

        if filename:
            pd.DataFrame(data, columns=['Name', 'Min', 'Max', 'Mean', 'Total']).to_csv(filename, index=False)

    def reset(self):
        self.time_tracker.clear()
        self.phase = ''

def main():
    phase = 'PHASE ONE'
    for _ in range(1):
        with Timer(phase):
            time.sleep(.1)

    phase = 'PHASE TWO'
    for _ in range(2):
        with Timer(phase):
            time.sleep(.2)

    phase = 'PHASE THREE'
    for _ in range(3):
        with Timer(phase):
            time.sleep(.3)

    Timer().report_phases('timer_1.csv')

    Timer().reset()

    phase = 'PHASE FOUR'
    for _ in range(4):
        with Timer(phase) as timer:
            time.sleep(.4)

    Timer().report_phases('timer_2.csv')

if __name__ == '__main__':
    main()