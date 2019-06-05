from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
import os
import numpy as np


class AttrDict(dict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
          raise KeyError('Non-existent config key: {}'.format(full_key))

        v = _decode_cfg_value(v_)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def update_cfg(cfg):
    cfg.TRAIN.LR_SCHEDULER.MAX_EPOCHS = cfg.TRAIN.MAX_EPOCHS - cfg.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS
    cfg.DATASET.IMAGE_SIZE = cfg.MODEL.IMAGE_SIZE
    cfg.DATASET.TRAIN_BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
    cfg.DATASET.TEST_BATCH_SIZE = cfg.TEST.BATCH_SIZE
    cfg.MATCHER.NUM_CLASSES = cfg.MODEL.NUM_CLASSES
    cfg.POST_PROCESS.NUM_CLASSES = cfg.MODEL.NUM_CLASSES
    cfg.POST_PROCESS.BACKGROUND_LABEL = cfg.MATCHER.BACKGROUND_LABEL
    cfg.POST_PROCESS.VARIANCE = cfg.MATCHER.VARIANCE
    cfg.CHECKPOINTS_PREFIX = '{}_{}_{}'.format(cfg.MODEL.SSDS, cfg.MODEL.NETS, cfg.DATASET.DATASET)
    return cfg

def _read_default_config():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.join(this_dir, "..")
    configs_dir = os.path.join(root_dir, "cfgs", "default")

    train_cgf_path = os.path.join(configs_dir, "train.yaml")
    test_cgf_path = os.path.join(configs_dir, "test.yaml")
    model_cgf_path = os.path.join(configs_dir, "model.yaml")

    def cast_dict(d):
        if isinstance(d, dict):
            for x in d:
                d[x] = cast_dict(d[x])
            return AttrDict(d)
        else:
            try:
                return literal_eval(d)
            except Exception:
                return d


    import yaml
    with open(train_cgf_path, 'r') as f:
        train_cgf = AttrDict(yaml.load(f))
        train_cgf = cast_dict(train_cgf)
    with open(test_cgf_path, 'r') as f:
        test_cgf = AttrDict(yaml.load(f))
        test_cgf = cast_dict(test_cgf)
    with open(model_cgf_path, 'r') as f:
        model_cgf = AttrDict(yaml.load(f))
        model_cgf = cast_dict(model_cgf)

    z = AttrDict({**train_cgf, **test_cgf})
    out = AttrDict({**z, **model_cgf})
    out.ROOT_DIR = root_dir
    out.EXP_DIR = os.path.join(root_dir, 'experiments/models/')
    return out


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))

    cfg = _read_default_config()

    _merge_a_into_b(yaml_cfg, cfg)
    cfg = update_cfg(cfg)
    return cfg

def get_config(name):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.join(this_dir, "..")
    cfg = os.path.join(root_dir, "cfgs", f"{name}.yaml")
    return cfg_from_file(cfg)

def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v

def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a