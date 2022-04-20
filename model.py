"""
Created on Thu Jan  6 15:21:18 2022

@author: linh.trinh

this is intraday model code to run  
"""

import pandas as pd
import numpy as np
from optparse import OptionParser
import logging

### LIBS & CONFIGS

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quantlib.quantlib_id import tsOp, utilOp, simOp
from config.config_id import model_config
# from quantlib_id import tsOp, utilOp, simOp
# from config_id import model_config 
config = model_config.Model_Config()

def generate_alpha(h5_return_df, product_id, generic_no, parameters_dict, start_date = None, end_date = None):
    
    h5_return_df = h5_return_df.copy()
    return_df = utilOp.get_single_instrument_di_ti_return_dataframe(product_id, generic_no, h5_return_df, parameters_dict)
    dates_v = np.array(return_df.index)
    return_m = return_df.to_numpy().copy()
    time_m = np.array([return_df.columns] * len(return_df))
    
    # get parameters for product 
    start_signal_time = parameters_dict[product_id]['start_signal_time']
    end_signal_time = parameters_dict[product_id]['end_signal_time']
    lookback_period = parameters_dict[product_id]['lookback_period']
    z_score_threshold = parameters_dict[product_id]['z_score_threshold']
    start_trade_time = parameters_dict[product_id]['start_trade_time']
    end_trade_time = parameters_dict[product_id]['end_trade_time']
    
    # calculate intraday return
    signal_return_v = utilOp.get_intraday_return(return_m, time_m, start_signal_time, end_signal_time)
    signal_return_v = signal_return_v[:,np.newaxis]
    
    # compare intraday return to historical
    trend_zscore_v = (signal_return_v - tsOp.movingAverage(signal_return_v, lookback_period))/ np.power(tsOp.movingVariance(signal_return_v, lookback_period), 0.5)
    trend_zscore_v = trend_zscore_v.flatten()
    
    # alpha signal: following intraday return when it is strong
    alpha_v = np.sign(signal_return_v.flatten())
    alpha_v[np.bitwise_and(alpha_v > 0, trend_zscore_v < z_score_threshold)] = 0
    alpha_v[np.bitwise_and(alpha_v < 0, trend_zscore_v > -z_score_threshold)] = 0
    alpha_v[~np.isfinite(alpha_v)] = 0
    
    alpha_df = pd.DataFrame(data = alpha_v, index = dates_v)
    
    # realized intraday return
    trade_return_v = utilOp.get_intraday_return(return_m, time_m, start_trade_time, end_trade_time)
    trade_return_df = pd.DataFrame(data = trade_return_v, index = dates_v)
    
    if start_date:
        alpha_df = alpha_df.loc[alpha_df.index >= int(start_date)]
        trade_return_df = trade_return_df.loc[trade_return_df.index >= int(start_date)]
    if end_date:
        alpha_df = alpha_df.loc[alpha_df.index <= int(end_date)]
        trade_return_df = trade_return_df.loc[trade_return_df.index <= int(end_date)]
    
    return alpha_df, trade_return_df

class Model:
    
    def __init__(self, start_date, end_date, back_test):
        self.start_date = start_date
        self.end_date = end_date
        self.back_test = back_test
        self.product_ids_list_to_trade = config.product_ids_list_to_trade
        self.parameters_dict = config.product_parameters_dict
        self.contract_code_map = config.contract_code_map
        self.data_folder = config.data_dir_location
        self.output_folder = config.output_dir_location
        self.logger = logging.getLogger(__name__)
        
    def read_data(self):
        self.logger.info('Start reading HDF5 30-minute bar data on {}'.format(self.end_date))
        self.input_hdf5_file = self.data_folder + 'QF2_30min_1100_' + self.end_date + '.h5'
        self.return_df = pd.read_hdf(self.input_hdf5_file, 'returns_lc' )
        self.price_df = pd.read_hdf(self.input_hdf5_file, 'prices_lc')
        self.underlying_instr_id_df = pd.read_hdf(self.input_hdf5_file, 'underlying_instr_id')
        self.underlying_instr_code_df = pd.read_hdf(self.input_hdf5_file, 'underlying_instr_code')
        self.underlying_instr_code_str_df = self.underlying_instr_code_df.applymap(lambda x: utilOp.transform_num_to_instr_code(x, self.contract_code_map))
    
    def generate_trades(self):
        self.logger.info('Start generating trades on {}'.format(self.end_date))
        output_trades_list = []
        alpha_matrix = []
        pnl_matrix = []
        for product_id in self.product_ids_list_to_trade:
            generic_list = self.parameters_dict[product_id]['generic_to_trade']
            for i in range(len(generic_list)):
                generic_no = generic_list[i]
                qf2_id = utilOp.get_qf2_id(product_id, generic_no, self.parameters_dict)
                self.logger.info('Generating trade for {}'.format(qf2_id))
                
                # alpha generation
                alpha_df, trade_return_df = generate_alpha(self.return_df, product_id, generic_no, self.parameters_dict, self.start_date, self.end_date)
                alpha_df.columns = [qf2_id]
                alpha_matrix.append(alpha_df)
                
                # backtest stats and plot
                if self.back_test == True:
                    TC_c_bps = self.parameters_dict[product_id]['TC_c_bps']
                    TC_ba_bps = self.parameters_dict[product_id]['TC_ba_bps']
                    pnl_v, _ = simOp.sim(alpha_df.values, trade_return_df.values, TC_c_bps, TC_ba_bps)
                    simOp.show_strategy_plot(pnl_v, alpha_df.index, qf2_id)
                    pnl_df = pd.DataFrame(data = pnl_v, columns = alpha_df.columns, index = alpha_df.index)
                    pnl_matrix.append(pnl_df)
                
                # calculate lots to trade
                # price = self.price_df[[qf2_id]].iloc[-1].values[0]
                # multiplier = self.parameters_dict[product_id]['contract_multiplier']
                # alloc = self.parameters_dict[product_id]['alloc_in_millions'][i]
                # lots_to_trade = np.sign(alpha_df.loc[int(self.end_date)].values[0]) * (alloc * 1e6) / (price * multiplier)
                lots_to_trade = self.parameters_dict[product_id]['alloc_in_lots'][i] * np.sign(alpha_df.loc[int(self.end_date)].values[0])
                lots_to_trade = np.round(lots_to_trade)
                
                # get reference data such as instrument id, identifier
                instrument_id = self.underlying_instr_id_df[[qf2_id]].iloc[-1].values[0]
                bbg_id = self.parameters_dict[product_id]['bbg_id']
                bbg_ylw_key = self.parameters_dict[product_id]['bbg_ylw_key']
                identifier = bbg_id + self.underlying_instr_code_str_df[[qf2_id]].iloc[-1].values[0] + ' ' + bbg_ylw_key
                
                if lots_to_trade > 0: output_trades_list.append([instrument_id, identifier, lots_to_trade])
        output_trades_arr = np.array(output_trades_list)
        if len(output_trades_arr) > 0:
            self.output_trade_df = pd.DataFrame(output_trades_arr, columns = ['instrument_id','identifier','lots_to_trade'])
        else:
            self.output_trade_df = pd.DataFrame(columns = ['instrument_id','identifier','lots_to_trade'])
        self.alpha_matrix = pd.concat(alpha_matrix, axis=1)
        self.pnl_matrix = pd.concat(pnl_matrix, axis=1)
        
    def write_trades(self):
        output_df = self.output_trade_df
        file_name = self.output_folder + 'Intraday_Trade_File_for_' + self.end_date + '.xlsx'
        output_df.to_excel(file_name, index = False)
        
    def save_alpha_to_hdf5(self):
        output_dest = self.output_folder + 'intraday_alpha_' + self.end_date + '.h5'
        self.alpha_matrix.to_hdf(output_dest, key='alpha_matrix', format='fixed')
        self.pnl_matrix.to_hdf(output_dest, key='pnl_matrix', format='fixed')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-s', '--start_date', help = 'start_date for model, format: yyyymmdd', type = str, default = '20120102')
    parser.add_option('-e', '--end_date', help = 'end_date for model, format: yyyymmdd', type = str)
    parser.add_option('-t', '--back_test', help = 'option to run back-test or not', type = int, default = True)
    parser.add_option('-w', '--write_to_excel', help = 'option to write trade to excel file or not', type = int, default = True)
    
    (options, args) = parser.parse_args()
    
    start_date = options.start_date
    end_date = options.end_date
    start_date = '20211215'
    end_date = '20220316'
    if end_date is None:
        end_date = (pd.datetime.today()).date().strftime('%Y%m%d')
    back_test = options.back_test
    write_to_excel = options.write_to_excel
    
    print('Model Start Date: ', start_date)
    print('Model End Date: ', end_date)
    print('Run backtest (T/F): ', back_test)
    print('Write to excel file: ', write_to_excel)
    
    model = Model(start_date, end_date, back_test)
    model.read_data()
    model.generate_trades()
    if write_to_excel:
        model.write_trades()
    model.save_alpha_to_hdf5()
    
    
    
    
