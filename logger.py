import os
import time
import logging


def get_logger(args, log_name):
    # 返回当地的时间
    record_time = time.strftime('%Y%m%d-%H:%M', time.localtime(time.time()))
    logger_save_path = args.log_path + args.data_name + '/logs/'
    # 创建目录
    if not os.path.exists(logger_save_path):
        os.makedirs(logger_save_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)  # 设置为INFO级别
    if log_name:
        handler = logging.FileHandler(logger_save_path + f'/{log_name}.log')
    else:
        handler = logging.FileHandler(logger_save_path + f'/{args.model_name}_' + record_time + '.log')
    handler.setLevel(logging.INFO)
    # 创建一个格式器（formatter）
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 作用在handler上
    handler.setFormatter(formatter)
    # 添加处理器
    logger.addHandler(handler)

    return logger
