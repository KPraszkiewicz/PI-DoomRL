import configparser


def save_config(config_name=""): #TODO
    if config_name == "":
        return

    config = configparser.ConfigParser()

    with open(config_name, 'w') as configfile:
        config.write(configfile)


def load_config(config_name="config.ini"):
    config = configparser.ConfigParser()
    config.read(config_name)

    for section in config.sections():
        for key in config[section]:
            print(f"{section}: {key} - {config[section][key]}")

    return config
