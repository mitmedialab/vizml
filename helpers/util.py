from time import strftime, localtime


def get_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())
