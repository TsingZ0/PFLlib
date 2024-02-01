import subprocess
import random
import time
from collections import deque
from enum import Enum
from gpu_utils import get_gpu_mem_info, get_cpu_mem_info

# Enum class describing the status of the program
class ProgramStatus(Enum):
    NOT_RUN = 1
    RUNNING = 2
    FINISHED = 3
    FAILED = 4

# A program instance representing a single ML training command
class ProgramInstance:
    def __init__(self, command):
        self.command = command
        self.status = ProgramStatus.NOT_RUN
        
    def run(self):
        print(f"Running command: {self.command}")
        self.status = ProgramStatus.RUNNING
        try:
            # Simulating a command execution
            time.sleep(random.random() * 2)  # Simulate execution time
            # Uncomment the following line to run the actual command
            # subprocess.check_output(self.command, shell=True)
            
            # Simulate memory check, sometimes failing the job
            if random.random() > 0.7:
                raise MemoryError("Out of memory exception")
            
            self.status = ProgramStatus.FINISHED
        
        except (subprocess.CalledProcessError, MemoryError) as e:
            print(f"Execution failed: {e}")
            self.status = ProgramStatus.FAILED
            time.sleep(600)  # Wait for 10 minutes
            raise e
        print(f"Finished command: {self.command}")

# The Launcher responsible for running Programs
class Launcher:
    def __init__(self, commands):
        self.waiting_queue = deque(ProgramInstance(cmd) for cmd in commands)
        self.running_queue = deque()
    
    def run_programs(self):
        while True:
            while self.waiting_queue:
                prog = self.waiting_queue.popleft()
                try:
                    prog.run()
                    self.running_queue.append(prog)
                except Exception as e:
                    self.waiting_queue.appendleft(prog)
            
            time.sleep(1)  # Check the status every second
            
            if not self.waiting_queue and all(p.status == ProgramStatus.FINISHED for p in self.running_queue):
                print("All programs have finished running.")
                break
            elif all(p.status in (ProgramStatus.FINISHED, ProgramStatus.FAILED) for p in self.running_queue):
                print("Some programs failed but all have stopped running.")
                break

orders = [
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedProx -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 > cifar10_fedProx_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedFomo -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 > cifar10_fedFomo_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo Ditto -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_ditto_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedALA -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedALA_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo GPFL -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_GPFL_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedPAC -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedPAC_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo MOON -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_MOON_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedAvg -gr 100 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedAvg_dir.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo pFedMe -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -bt 1 -lam 15 -ls 5 --partition pat > cifar10_pFedMe_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedProx -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 --partition pat > cifar10_fedProx_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedFomo -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 --partition pat > cifar10_fedFomo_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo Ditto -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_ditto_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedALA -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedALA_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo GPFL -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat  > cifar10_GPFL_pat.out 2>&1 &",
    "nohup python -u main.py -data Cifar10 -m cnn -algo FedPAC -gr 100 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedPAC_pat.out 2>&1 &"
]
# Example usage
if __name__ == "__main__":
    ml_programs = orders  # Your list of commands
    launcher = Launcher(ml_programs)
    launcher.run_programs()