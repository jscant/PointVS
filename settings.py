"""Handy settings class, useful for on-the-fly modification of config files."""

from pathlib import Path

import yaml
from lie_conv.lieGroups import T, SE3, SO3


class Settings:
    """This class can modify yaml settings files on-the fly. Usage:

    with Settings(load_loc, save_loc) as s:
        s.setting_1 = value

    This will load the settings from the yaml file load_loc, filling in any
    missing defaults with those found in Settings._default_settings, saving
    any modifications to save_loc. If load_loc is None, no settings are loaded
    and only defaults are used. If save_loc is None, the yaml file found at
    load_loc is overwritten, including modifications in the `with` block.
    """

    _default_settings = {}
    _remap_on_save = {}
    _ignore_on_save = []
    _ignore_on_report = ['group_name']

    def __init__(self, load_loc=None, save_loc=None):
        """Set up a yaml file associated with the instance of Settings.

        Arguments:
            load_loc: where to load settings from. If unspecified, uses all
                defaults found in _default_settings
            save_loc: where to store new file. If not specified, creates a new
                yaml file called model.yaml
        """
        if load_loc is not None:
            self.load_loc = Path(load_loc).expanduser()
            if self.load_loc.is_file():
                with open(self.load_loc) as f:
                    self.__dict__ = yaml.full_load(f)
            else:
                raise FileNotFoundError(
                    'Config file {} not found.'.format(load_loc))

        for key, value in self._default_settings.items():
            try:
                self.__dict__[key]
            except KeyError:
                self.__dict__[key] = value

        if save_loc is None:
            if load_loc is None:
                self._save_loc = Path('model.yaml')
            else:
                self._save_loc = load_loc
        else:
            self._save_loc = Path(save_loc).expanduser()
        self._save_loc.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        """Dumps any changes into the yaml file associated with the instance."""
        _settings = {key: value for key, value in self.__dict__.items() if key
                     not in self._ignore_on_save + ['_save_loc', 'load_loc']}
        for key, value in self._remap_on_save.items():
            _settings.update({key: getattr(self, value)})
        with open(self._save_loc, 'w') as f:
            yaml.dump(_settings, f)

    @property
    def settings(self):
        """Returns all settings as a dictionary."""
        return {key: value for key, value in self.__dict__.items() if
                key not in self._ignore_on_report + ['_save_loc', 'load_loc']}


class LieConvSettings(Settings):
    _group_map = {'T': T, 'SE3': SE3, 'SO3': SO3}

    with open('config/lieconv_conf.yaml') as f:
        _default_settings = yaml.full_load(f)

    _remap_on_save = {'group': 'group_name'}
    _ignore_on_save = ['group', 'group_name']
    _ignore_on_report = ['group_name']

    def __init__(self, load_loc=None, save_loc=None):
        super().__init__(load_loc=load_loc, save_loc=save_loc)
        self.group_name = self.group
        self.group = self._group_map[self.group]()


class SE3TransformerSettings(Settings):
    with open('config/se3trans_conf.yaml') as f:
        _default_settings = yaml.full_load(f)

    _remap_on_save = {}
    _ignore_on_save = []
    _ignore_on_report = []


class SessionSettings(Settings):
    with open('config/default_session.yaml') as f:
        _default_settings = yaml.full_load(f)
