import os
import yaml
from types import SimpleNamespace


def load_yaml_config(path: str, prefix_keys=False) -> SimpleNamespace:
    """
    Load a yaml configuration file from the specified path, apply preprocessing operations to it and return
    the configuration in a SimpleNamespace.

    :param path: Path to the configuration file.
    :return: SimpleNamespace containing the config.
    """
    with open(path) as f:
        config = SimpleNamespace(**yaml.load(f, Loader=yaml.FullLoader))
    config = preprocess_yaml_config(config, prefix_keys=prefix_keys)
    return config


def preprocess_yaml_config(config: SimpleNamespace, prefix_keys=False) -> SimpleNamespace:
    """
    Preprocess a simple namespace. Currently,
    - prepend the prefix key to all the configuration parameters
    - change 'None' strings to None values
    :param config: The SimpleNamespace containing the configuration.
    :return: Preprocessed configuration as a SimpleNamespace
    """
    # Make sure there's a prefix in the configuration
    assert 'prefix' in config.__dict__, 'Please include a prefix in the yaml.'

    if prefix_keys:
        # Grab the prefix from the yaml file
        prefix = config.prefix

        # Prepend the prefix to all the keys, and get rid of the prefix
        config = SimpleNamespace(**{f'{prefix}_{k}': v for k, v in config.__dict__.items() if k != prefix})

    # Change 'None' to None in the top level: recommended behavior is to use null instead of None in the yaml
    for key, value in config.__dict__.items():
        config.__dict__[key] = value if value != 'None' else None

    return config


def subtract_simple_namespaces(sns_main: SimpleNamespace, sns_diff: SimpleNamespace) -> SimpleNamespace:
    """
    Subtract a SimpleNamespace from another. Subtraction corresponds to removing keys in sns_main that are present
    in sns_diff.
    :param sns_main: The SimpleNamespace to subtract *from*.
    :param sns_diff: The SimpleNamespace to subtract off.
    :return: A SimpleNamespace containing the difference.
    """
    # Find the keys that are in sns_main but not in sns_diff
    diff_keys = sns_main.__dict__.keys() - sns_diff.__dict__.keys()

    # Return a SimpleNamespace that contains the diff_keys
    return SimpleNamespace(**{k: sns_main.__dict__[k] for k in diff_keys})


def update_simple_namespace(sns_main: SimpleNamespace, sns_added: SimpleNamespace) -> SimpleNamespace:
    """
    Update a SimpleNamespace with another and return the updated SimpleNamespace. For keys that overlap,
    sns_added's values will replace the original values in sns_main.
    :param sns_main: The SimpleNamespace that is updated.
    :param sns_added: The SimpleNamespace that is added in.
    :return: An updated SimpleNamespace.
    """
    # Create a new SimpleNamespace which contains the data in sns_main
    updated_sns = SimpleNamespace(**sns_main.__dict__)

    # Update this SimpleNamespace with data from sns_added
    updated_sns.__dict__.update(sns_added.__dict__)

    return updated_sns


def recursively_create_config_simple_namespace(config_path, base_template_config_path, match_on='prefix'):
    """
    A helper function to create a config SimpleNamespace that can be passed to train methods. Requires a
    config and its template, and replaces the template with the settings you've specified.
    """

    def _update_config(_config, _template_config):
        # Make sure that config and parent are matched
        if match_on is not None:
            assert _config.__dict__[match_on] == _template_config.__dict__[match_on], \
                f'Configs are mismatched {_config.__dict__[match_on]} =/= {_template_config.__dict__[match_on]}.'

        # Make sure we didn't include any configuration options that aren't in the parent
        extra_keys = subtract_simple_namespaces(_config, _template_config).__dict__
        assert len(extra_keys) == 0, f'Extra configuration option specified in {config_path}: {extra_keys}.'

        # Update the template with the configuration options from the config
        _config = update_simple_namespace(_template_config, _config)

        return _config

    def _recurse(_config, _base_template_config):
        assert 'parent_template' in _config.__dict__, 'The parent_template argument must be implemented.'

        # Find the location of the parent configuration file
        if _config.parent_template is None:
            # There's no parent template: this config points to the base_template_config
            parent_config_path = base_template_config_path
        else:
            # There's a parent template: we assume that it must be in the same folder as the config
            parent_config_path = os.path.join(os.path.dirname(config_path), _config.parent_template)

        # Load up the parent config
        parent_config = load_yaml_config(parent_config_path)
        parent_config = _update_config(parent_config, _base_template_config)

        # The template the parent points to
        parent_template = parent_config.parent_template

        # Update the config using the parent's
        _config = _update_config(_config, parent_config)

        # Add the parent's path to the list of applied templates
        _config._template_config_path.append(parent_config_path)

        if _config.parent_template is None:
            # Return if the parent template is None
            return _config

        # Replace the template parameter with the parent's: now if we recurse we'll be using the parent's template
        # to do another update
        _config.parent_template = parent_template

        # Recurse and return the config
        return _recurse(_config, _base_template_config)

    # Load up the config files
    config = load_yaml_config(config_path)  # top level config
    base_template_config = load_yaml_config(base_template_config_path)  # base template

    # Keep track of what templates are applied to the config, and where this config is
    config._template_config_path = []
    config._config_path = config_path

    # Remember who this config's parent is
    if 'parent_template' not in config.__dict__:
        config.parent_template = None
    parent = config.parent_template

    # Recurse to apply all the parent configurations
    config = _recurse(config, base_template_config)

    # Ensure the parent is set correctly
    config.parent_template = parent

    # Assert to make sure we hit the base template config
    assert config._template_config_path[-1] == base_template_config_path, '{template_config_path} is never used.'

    return config


def create_config_simple_namespace(config_path, template_config_path, match_on='prefix'):
    """
    A helper function to create a config SimpleNamespace that can be passed to train methods. Requires a
    config and its template, and replaces the template with the settings you've specified.
    """
    # Load up the config files
    config = load_yaml_config(config_path)
    template_config = load_yaml_config(template_config_path)

    # Make sure that config and template are matched
    if match_on is not None:
        assert config.__dict__[match_on] == template_config.__dict__[match_on], \
            f'Configs are mismatched {config.__dict__[match_on]} =/= {template_config.__dict__[match_on]}.'

    # Make sure we didn't include any configuration options that aren't in the template
    extra_keys = subtract_simple_namespaces(config, template_config).__dict__
    assert len(extra_keys) == 0, \
        f'Extra configuration option specified: {extra_keys}'

    # Update the template with the configuration options that we picked
    config = update_simple_namespace(template_config, config)

    # Add the config and template path to the config
    config._config_path = config_path
    config._template_config_path = template_config_path

    return config


def pretty_print_simple_namespace(sns):
    """
    Pretty print a SimpleNamespace. Currently just loops over and prints each (key, value) pair.
    """
    # Loop and print the outer level
    for k, v in sns.__dict__.items():
        print(f'{k}: {v}')
