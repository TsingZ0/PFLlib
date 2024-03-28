import subprocess
import random
import time
from collections import deque
from enum import Enum
from gpu_utils import get_gpu_mem_info, get_cpu_mem_info, is_available
import os
import signal
import psutil

# Enum class describing the status of the program
class ProgramStatus(Enum):
    NOT_RUN = 1
    RUNNING = 2
    FINISHED = 3
    FAILED = 4

# A program instance representing a single ML training command
class ProgramInstance:
    def __init__(self, command, virtual_env):
        self.command = f"{virtual_env} {command}"
        self.status = ProgramStatus.NOT_RUN
        self.pid = 0
        self.proc: subprocess.Popen = None
        self.wrong_num = 0 #后期可以进行一个超时检测
        
    def run(self):
        if not is_available():
            print("当前内存或显存占用过高，撤销运行"+self.command)
            return -1
        else:
            print(f"Running command: {self.command}")
            self.status = ProgramStatus.RUNNING
            proc = subprocess.Popen(self.command, shell=True, stdout=subprocess.PIPE)
            print(self.command+"'s process id is:", proc.pid)
            self.proc = proc
            self.pid = proc.pid
            return 0

# The Launcher responsible for running Programs
class Launcher:
    def __init__(self, commands, virtual_env):
        self.waiting_queue = deque(ProgramInstance(cmd, virtual_env) for cmd in commands)
        self.running_queue = deque()

    def check_process_state(self, prog: ProgramInstance):
        status = prog.proc.poll()
        return status
    
    def run_programs(self):
        while True:
            if self.waiting_queue or self.running_queue:
                prog:ProgramInstance = self.waiting_queue.popleft()
                try:
                    rt = prog.run()
                    if rt == -1:
                        self.waiting_queue.appendleft(prog)
                    else:
                        self.running_queue.append(prog)
                        time.sleep(15)
                except Exception as e:
                    print(str(e))
            
            if len(self.waiting_queue)==0 and len(self.running_queue)==0:
                print("所有任务运行完成")

            # 更新进程状态
            need_del = []
            for prog in self.running_queue:
                status = prog.proc.poll()
                print(f"{prog.command}:", status)
                if status == None: continue
                elif status >= 0:
                    print(f"{prog.command} 已经运行完成")
                    prog.status = ProgramStatus.FINISHED
                    need_del.append(prog)
                else:
                    print(prog.command+"未正常结束，重试")
                    self.waiting_queue.append(prog)
                    prog.status = ProgramStatus.NOT_RUN

            for prog in need_del:
                self.running_queue.remove(prog)

            if is_available():
                time.sleep(5)
                pass
            else:
                time.sleep(5)  #基本上一个批次在半个小时左右，但这里以 5 分钟 (300)为单位进行轮询

# orders不要加nohup！！！否则无法追踪子进程
orders = [
    "-u main.py -data Cifar10 -m cnn -algo FedProx -gr 3 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 > cifar10_fedProx_dir.out",
    "-u main.py -data Cifar10 -m cnn -algo FedFomo -gr 3 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 > cifar10_fedFomo_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo Ditto -gr 3 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_ditto_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedALA -gr 3 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedALA_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo GPFL -gr 3 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_GPFL_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedPAC -gr 3 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedPAC_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo MOON -gr 20 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_MOON_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedAvg -gr 20 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedAvg_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo pFedMe -gr 20 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -bt 1 -lam 15 -ls 5 --partition pat > cifar10_pFedMe_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedProx -gr 20 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 --partition pat > cifar10_fedProx_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedFomo -gr 20 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 --partition pat > cifar10_fedFomo_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo Ditto -gr 20 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_ditto_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedALA -gr 20 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedALA_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo GPFL -gr 20 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat  > cifar10_GPFL_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedPAC -gr 20 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedPAC_pat.out 2>&1 &"
]
# Example usage
if __name__ == "__main__":
    ml_programs = orders  # Your list of commands
    virtual_env = "~/miniconda3/envs/fl/bin/python"
    launcher = Launcher(ml_programs, virtual_env)
    launcher.run_programs()