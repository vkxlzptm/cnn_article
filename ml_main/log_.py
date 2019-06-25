

import datetime
import traceback
LOG_PATH = "log/"

def status_logging(s):
    with open(LOG_PATH+'status_log.txt','a') as f:
        f.write(str(datetime.datetime.now())+' '+s+'\n')
    return 
def error_logging():
    s = str(datetime.datetime.now()) + ' ' + traceback.format_exc() + '\n'
    with open(LOG_PATH+'error_log.txt','a') as f:
        f.write(s+'\n')
    return


