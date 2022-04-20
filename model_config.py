"""
Created on Thu Jan  6 15:21:18 2022

@author: linh.trinh

this is intraday model parameters config
"""

__docformat__ = 'restructuredtext en'

class Model_Config(object):
    
    # general parameteres
    contract_code_map = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
    data_dir_location = 'H:/Business Team/PF2/Linh/QF2-HDF5/'
    output_dir_location = 'H:/Business Team/PF2/Linh/model_output/'
    # data_dir_location = 'E:/QF/inv_qf2/App/Git/matlab/QF2-data/QF2-HDF5/'
    # output_dir_location = 'E:/QF/inv_qf2/App/Git/matlab/QF2-data/QF2-output/model_output'
    
    # strategy parameters
    product_ids_list_to_trade = [1, 82]
    
    product_parameters_dict = {
        1: {'start_signal_time': -1500, 'end_signal_time': 600, 'start_trade_time': 800, 'end_trade_time': 1430, 'timezone': 'America/New_York', 
            'lookback_period': 60, 'z_score_threshold': 0.75, 'TC_c_bps': 0.40, 'TC_ba_bps': 2.30, 'generic_to_trade': [1], 'alloc_in_lots': [5],
            'contract_multiplier': 1000, 'bbg_id': 'CL', 'bbg_ylw_key': 'Comdty', 'NB_slct': [i for i in range(1, 4)]},
        82: {'start_signal_time': -2000, 'end_signal_time': 1100, 'start_trade_time': 1300, 'end_trade_time': 1930, 'timezone': 'Europe/London', 
            'lookback_period': 60, 'z_score_threshold': 0.75, 'TC_c_bps': 0.20, 'TC_ba_bps': 1.85, 'generic_to_trade': [1], 'alloc_in_lots': [5],
            'contract_multiplier': 1000, 'bbg_id': 'CO', 'bbg_ylw_key': 'Comdty', 'NB_slct': [i for i in range(1, 4)]},
    }

