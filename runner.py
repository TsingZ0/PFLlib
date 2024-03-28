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
        self.proc = None
        self.wrong_num = 0 #后期可以进行一个超时检测
        
    def run(self):
        print(f"Running command: {self.command}")
        self.status = ProgramStatus.RUNNING
        proc = subprocess.Popen(self.command, shell=True, stdout=subprocess.PIPE)
        time.sleep(30)  # 估计程序启动时间
        print(self.command+"'s process id is:", proc.pid)
        tmp = is_available()
        if not is_available():
            os.kill(proc.pid, signal.SIGINT)
            print("当前内存或显存占用过高，撤销运行"+self.comman)
            self.status = ProgramStatus.NOT_RUN
        else:
            self.proc = proc
            self.pid = proc.pid

# The Launcher responsible for running Programs
class Launcher:
    def __init__(self, commands, virtual_env):
        self.waiting_queue = deque(ProgramInstance(cmd, virtual_env) for cmd in commands)
        self.running_queue = deque()

    def check_process_state(self, prog):
        status = prog.proc.poll()
        return status
    
    def run_programs(self):
        while True:
            if self.waiting_queue or self.running_queue:
                prog = self.waiting_queue.popleft()
                try:
                    prog.run()
                    if prog.pid == 0:
                        self.waiting_queue.appendleft(prog)
                    else:
                        self.running_queue.append(prog)
                except Exception as e:
                    print(str(e))
            
            if not self.waiting_queue and all(p.status == ProgramStatus.FINISHED for p in self.running_queue):
                print("All programs have finished running.")
                break
            elif all(p.status in (ProgramStatus.FINISHED, ProgramStatus.FAILED) for p in self.running_queue):
                print("Some programs failed but all have stopped running.")
                break

            # 更新进程状态
            need_del = []
            for prog in self.running_queue:
                status = self.check_process_state(prog)
                print("status: ",status)
                if status == None: pass
                elif status >= 0:
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
                time.sleep(300)  #基本上一个批次在半个小时左右，但这里以 5 分钟为单位进行轮询

# orders不要加nohup！！！否则无法追踪子进程
orders = [
    "-u main.py -data Cifar10 -m cnn -algo FedProx -gr 2 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 > cifar10_fedProx_dir.out",
    "-u main.py -data Cifar10 -m cnn -algo FedFomo -gr 2 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 > cifar10_fedFomo_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo Ditto -gr 2 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_ditto_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedALA -gr 2 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedALA_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo GPFL -gr 2 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_GPFL_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedPAC -gr 2 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedPAC_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo MOON -gr 2 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_MOON_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedAvg -gr 2 -did 0 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 > cifar10_FedAvg_dir.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo pFedMe -gr 2 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -bt 1 -lam 15 -ls 5 --partition pat > cifar10_pFedMe_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedProx -gr 2 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -mu 0.001 --partition pat > cifar10_fedProx_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedFomo -gr 2 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 -M 5 --partition pat > cifar10_fedFomo_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo Ditto -gr 2 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_ditto_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedALA -gr 2 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedALA_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo GPFL -gr 2 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat  > cifar10_GPFL_pat.out 2>&1 &",
    "-u main.py -data Cifar10 -m cnn -algo FedPAC -gr 2 -did 1 -go cnn -lbs 64 -nc 10 -jr 1 -nb 10 -ls 5 --partition pat > cifar10_FedPAC_pat.out 2>&1 &"
]
# Example usage
if __name__ == "__main__":
    ml_programs = orders  # Your list of commands
    virtual_env = "~/miniconda3/envs/fl/bin/python"
    launcher = Launcher(ml_programs, virtual_env)
    launcher.run_programs()