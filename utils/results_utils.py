import numpy as np
import pandas as pd
import warnings


class BaseSubsamplingType():
    
    def __init__(self, dataset_name = None, label = None, color = None, marker = None):
        self.dataset_name = dataset_name
        self.name = self.get_name()
        self.label = label
        self.color = color
        self.marker = marker
        self.parameter_loader = self.set_parameter_loader()
        self.data_dict = self.create_data_dict()
        assert set(self.zipped_parameters()) < set(self.set_parameter_loader().keys()), "Some parameters are not in the parameter loader"
    
    def get_color(self):
        if self.color:
            return self.color
        else:
            raise NotImplementedError

    def get_marker(self):
        if self.marker:
            return self.marker
        else:
            raise NotImplementedError
        
    def get_label(self):
        if self.label:
            return self.label
        else:
            raise NotImplementedError
            
    def get_name(self):
        raise NotImplementedError
    
    def set_parameter_loader(self):
        raise NotImplementedError
    
    def zipped_parameters(self):
        raise NotImplementedError
    
    def create_data_dict(self):
        return {key:[] for key in self.parameter_loader.keys()} | {"run_name":[], "num_events":[], "bandwidth":[], "val_acc":[], "test_acc":[]}
    
    def get_parameters(self, config):
        return {key: value(config) for key, value in self.parameter_loader.items()}
    
    def add_bandwidth(self, run):
        num_events = None
        bandwidth = None
        if "all/num_events_mean" in run.summary and "all/bandwidth_mean" in run.summary:
            if "all/num_events_mean":
                num_events = run.summary["all/num_events_mean"]
            if "all/bandwidth_mean":
                bandwidth = run.summary["all/bandwidth_mean"]
        self.data_dict.setdefault("num_events", []).append(num_events)
        self.data_dict.setdefault("bandwidth", []).append(bandwidth)
    
    def add_accuracy(self,run):
        val_acc = None
        test_acc = None
        if 'transform' in run.config and 'train' in run.config['transform']:
            val_acc_key, test_acc_key = find_val_and_test_acc_keys(run)
        if val_acc_key:
            val_acc = run.summary[val_acc_key]
        if test_acc_key:
            test_acc = run.summary[test_acc_key]
        self.data_dict.setdefault("val_acc", []).append(val_acc)
        self.data_dict.setdefault("test_acc", []).append(test_acc)
        
    def process_run(self, run):
        if self.filter_run(run):
            config = run.config
            parameters = self.get_parameters(config)
            for key, value in parameters.items():
                self.data_dict[key].append(value)
            self.data_dict["run_name"].append(run.name)
            self.add_bandwidth(run)
            self.add_accuracy(run)
     
    def create_data_frames(self):
        self._convert_to_data_frame()
        self._agg_bandwidth_acc()
        self._agg_zipped_parameters()
        self._check_empty_num_events()
        self._clean_data_frame() 
        self. _assert_non_empty_num_events_in_zipped()
        
            
    def filter_run(self, run):
        raise NotImplementedError
    
    def _convert_to_data_frame(self):
        self.df = pd.DataFrame(self.data_dict)
        
    def _add_zipped_parameters(self): 
        self._zipped_parameters_label = '_'.join(self.zipped_parameters()) + '_zipped'
        self.df[self._zipped_parameters_label] = list(zip(*[self.df[z] for z in self.zipped_parameters()]))
    
    def _clean_data_frame(self):
        self.df = self.df.dropna(subset=["test_acc"])
        self.df_zipped = self.df_zipped.dropna(subset=[("test_acc","nanmean")])
        
    def _agg_zipped_parameters(self):
        self._add_zipped_parameters()
        self.df_zipped = self.df.groupby(
                by=self._zipped_parameters_label,
                dropna=False, 
                as_index=False).agg(
                    {
                        'val_acc': [np.nanmean, np.nanstd], 
                        'test_acc': [np.nanmean, np.nanstd],
                        'bandwidth': [np.nanmean, np.nanstd],
                        'num_events': [np.nanmean, np.nanstd],
                        
                    }
                )
    
    def _agg_bandwidth_acc(self):
        self.df = self.df.groupby(
                by=list(self.set_parameter_loader().keys()), 
                dropna=False, 
                as_index=False).agg(
                    {
                        'val_acc': np.nanmean, 
                        'test_acc': np.nanmean, 
                        'bandwidth': np.nanmean, 
                        'num_events': np.nanmean,
                    }
                )
    
    def _check_empty_num_events(self):
        if self.df['num_events'].isnull().any() or self.df['bandwidth'].isnull().any():
            self._fill_empty_num_events_with_mean()
            warnings.warn("Some 'num_events' or 'bandwidth are empty'!")
    
    def _assert_non_empty_num_events_in_zipped(self):
        if self.df_zipped['num_events']["nanmean"].isnull().any() or \
        self.df_zipped['bandwidth']["nanmean"].isnull().any():
            raise ValueError("Some 'num_events' are empty in zipped ")
        if self.df_zipped['num_events']["nanstd"].isnull().any() or \
        self.df_zipped['bandwidth']["nanstd"].isnull().any():
            warnings.warn("Std. Dev. of some 'num_events' are empty in zipped ")
                
    def _fill_empty_num_events_with_mean(self):
        warnings.warn("The empty 'num_events' are filled with the mean of the group")
        self.df['num_events'] = self.df.groupby(
            by=self._zipped_parameters_label,
            dropna=False,
            as_index=False)['num_events'].transform(lambda x: x.fillna(x.mean()))
        warnings.warn("The empty 'bandwidth' are filled with the mean of the group")
        self.df['bandwidth'] = self.df.groupby(
            by=self._zipped_parameters_label,
            dropna=False,
            as_index=False)['bandwidth'].transform(lambda x: x.fillna(x.mean()))
        
    def to_dict(self):
        return {
            'name': self.get_name(),
            'dataset_name': self.dataset_name,
            'df': self.df.to_json(),
            'df_zipped': self.df_zipped.to_json(),
        }
              
def find_val_and_test_acc_keys(run):
    val_acc_key = []
    test_acc_key = []
    for key in run.summary.keys():
        if "val" in key and "acc" in key and "mean" in key:
            val_acc_key.append(key)
        if "test" in key and "acc" in key and "mean" in key:
            test_acc_key.append(key)
    assert len(val_acc_key) <= 1, f"More than one val acc key found: {val_acc_key}"
    assert len(test_acc_key) <= 1, f"More than one test acc key found: {test_acc_key}"
    return val_acc_key[0] if len(val_acc_key) == 1 else None , test_acc_key[0] if len(test_acc_key) == 1 else None
    