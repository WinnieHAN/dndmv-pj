import subprocess
from multiprocessing import Pool
from config import parse_args
import sys


def strfind(a,b):
    return a[a.index(b):].split(' ')[1]


def Thread(arg):
    idx = int(strfind(arg, '-idx'))
    print('lalala')
    print('idx:  '+str(idx))
    cmd = arg[0:arg.index('-idx')]
    print('cmd:  '+cmd)
    javaOrPy = strfind(arg, '-javaOrPy')
    fname = "tuning1/idx-" + str(idx) + javaOrPy + ".log"
    file = open(fname, 'w')
    subprocess.call(cmd, shell=True, stdout=file)


def run_program_main(jv_arg, py_arg):
    # args = parse_args(type=1)
    # print(args)
    arglist = []
    # for i in range(1):
    jcmd = "/home/hanwj/software/jdk1.8.0_151/bin/java  -cp bin:\".:lib/*\"  yong.deplearning.GramLearnPy " + jv_arg
    arglist.append(jcmd)
    pcmd = "python pyFile/interface.py " + py_arg
    arglist.append(pcmd)

    p = Pool(2)
    p.map(Thread, arglist, chunksize=1)  # , chunksize=1
    p.close()
    p.join()

if __name__ == "__main__":
    # print('lalala')
    # print('sys.argv', sys.argv)
    # print(sys.argv[1])

    cmd = " ".join(sys.argv[1:])
    jp = cmd.split("SPLITTAG")
    jv_arg = jp[0]
    py_arg = jp[1]
    # print('sys.argv[1]' ,jv_arg)
    # print('sys.argv[2]' ,py_arg)
    run_program_main(jv_arg, py_arg)