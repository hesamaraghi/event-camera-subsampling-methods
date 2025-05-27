import torch_geometric.transforms as T
from datatransforms.event_transforms import *
from omegaconf import OmegaConf

def factory(cfg_all, transform_type = None, dataset_type = None):
      assert transform_type is not None, "transform_type must be provided"
      assert transform_type in ['transform', 'pre_transform'], "'transform_type' can only be 'transform' or 'pre_transform'"
      assert hasattr(cfg_all, transform_type), "'transform_type' not found in the config file" 
      assert dataset_type is not None, "dataset_type must be provided"
      assert dataset_type in ['train', 'validation', 'test'], f"{dataset_type} can only be 'train', 'validation' or 'test'"
      assert hasattr(getattr(cfg_all, transform_type), dataset_type), f"{dataset_type} not found in cfg.{transform_type}"
      cfg = getattr(getattr(cfg_all, transform_type), dataset_type)
      cfg_dict = OmegaConf.to_object(cfg)
      if 'transform' in cfg_dict and cfg.transform:
            transform_list = []
            if 'spatiotemporal_filtering_subsampling' in cfg_dict and cfg.spatiotemporal_filtering_subsampling.transform:
                  if 'mean_normalized' in cfg.spatiotemporal_filtering_subsampling and cfg.spatiotemporal_filtering_subsampling.mean_normalized:
                        transform_list.append(SpatioTemporalFilteringSubsamplingNormalized(cfg_all, cfg_dict)) 
                  else:
                        transform_list.append(SpatioTemporalFilteringSubsampling(cfg_all, cfg_dict))
            if 'tos_2DHarris_subsampling' in cfg_dict and cfg.tos_2DHarris_subsampling.transform:
                  if 'mean_normalized' in cfg.tos_2DHarris_subsampling and cfg.tos_2DHarris_subsampling.mean_normalized:
                        transform_list.append(TOS2DHarrisSubsamplingNormalized(cfg_all, cfg_dict)) 
                  else:
                        transform_list.append(TOS2DHarrisSubsampling(cfg_all, cfg_dict))
            if 'thresh_quantile' in cfg_dict and cfg.thresh_quantile is not None:
                  if 'spatiotemporal_filtering_subsampling' in cfg_dict:
                        assert not cfg.spatiotemporal_filtering_subsampling.transform, "thresh_quantile and spatiotemporal_filtering_subsampling cannot be used together"
                  transform_list.append(RemoveOutliers(cfg))
            if 'baseline_event_count' in cfg_dict and cfg.baseline_event_count.transform:
                  transform_list.append(BaselineEventCount(cfg_all, cfg_dict)) 
            if 'filter_nodes' in cfg_dict and cfg.filter_nodes is not None:
                  transform_list.append(FilterNodes(cfg))
            if 'shift_and_flip' in cfg_dict and cfg.shift_and_flip.transform:
                  transform_list.append(ShiftAndFlip(cfg.shift_and_flip))
            if 'temporal_scale' in cfg_dict and cfg.temporal_scale is not None:
                  transform_list.append(TemporalScaling(cfg))
            if 'temporal_quantization' in cfg_dict and cfg.temporal_quantization is not None:
                  transform_list.append(TemporalQuantization(cfg))
            if 'spatial_centering' in cfg_dict and cfg.spatial_centering:
                  transform_list.append(SpatialCentering())
            if 'spatial_subsampling' in cfg_dict and cfg.spatial_subsampling.transform:
                  transform_list.append(SpatialSubsampling( subsampling_ratios=(cfg.spatial_subsampling.h_r, cfg.spatial_subsampling.v_r),
                                                            subsampling_offsets=(cfg.spatial_subsampling.h_r_offset, cfg.spatial_subsampling.v_r_offset)))
            # if 'spatial_subsampling_random' in cfg_dict and cfg.spatial_subsampling_random.transform:
            #       transform_list.append(SpatialSubsamplingRandomOffset( cfg_transform=cfg_dict, subsampling_ratios=(cfg.spatial_subsampling_random.h_r, cfg.spatial_subsampling_random.v_r)))   
            if 'temporal_subsampling' in cfg_dict and cfg.temporal_subsampling.transform:
                  transform_list.append(TemporalSubsampling(subsampling_ratio=cfg.temporal_subsampling.subsampling_ratio, 
                                                            window_size=cfg.temporal_subsampling.window_size,
                                                            fixed_interval=cfg.temporal_subsampling.fixed_interval,
                                                            time_offset_coefficient=cfg.temporal_subsampling.time_offset_coefficient)) 
            if 'temporal_subsampling_random' in cfg_dict and cfg.temporal_subsampling_random.transform:
                  transform_list.append(TemporalSubsamplingRandomOffset( cfg_transform=cfg_dict, 
                                                                        subsampling_ratio=cfg.temporal_subsampling_random.subsampling_ratio, 
                                                                        window_size=cfg.temporal_subsampling_random.window_size,
                                                                        fixed_interval=cfg.temporal_subsampling_random.fixed_interval))      
            if 'random_ratio_subsampling' in cfg_dict and cfg.random_ratio_subsampling is not None:
                  transform_list.append(DropEventRandomly(cfg))    
            if 'num_events_per_sample' in cfg_dict and cfg.num_events_per_sample is not None:
                  if "fixed_sampling" in cfg_dict and cfg.fixed_sampling.transform is True:
                        transform_list.append(FixedSubsampling(cfg, replace = False, allow_duplicates = False))
                  else:
                        transform_list.append(T.FixedPoints(cfg.num_events_per_sample, replace = False, allow_duplicates = True))
            elif 'random_num_events_per_sample' in cfg_dict and cfg.random_num_events_per_sample.transform:
                  transform_list.append(VaryingSamplingPoints((   cfg.random_num_events_per_sample.min_num_events, 
                                                                  cfg.random_num_events_per_sample.max_num_events), 
                                                                  replace = False, 
                                                                  allow_duplicates = False,
                                                                  inverse_sampling = cfg.random_num_events_per_sample.inverse_sampling)) # TODO: allow duplicates is set to False for now
            if 'random_flip' in cfg_dict and cfg.random_flip is not None:
                  transform_list.append(T.RandomFlip(axis = 0, p = cfg.random_flip))
            if 'scale_limits' in cfg_dict and cfg.scale_limits is not None:
                  transform_list.append(SpatialScaling(cfg))
            if 'degree_limit' in cfg_dict and cfg.degree_limit is not None:
                  transform_list.append(T.RandomRotate(cfg.degree_limit, axis = 2))
            if 'radius_graph' in cfg_dict and cfg.radius_graph.transform:
                  transform_list.append(T.RadiusGraph(r = cfg.radius_graph.r,max_num_neighbors = cfg.radius_graph.max_num_neighbors))
            if 'radius_graph' in cfg_dict and cfg.radius_graph.add_edge_attr.transform:
                  transform_list.append(AddEdgeAttr(cfg.radius_graph.add_edge_attr))
            return T.Compose(transform_list)
      else:
            return None
      