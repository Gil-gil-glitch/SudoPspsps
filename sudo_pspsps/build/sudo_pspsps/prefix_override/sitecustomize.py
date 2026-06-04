import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ri-one/ri_one_master_ws/SudoPsPsPs/install/sudo_pspsps'
