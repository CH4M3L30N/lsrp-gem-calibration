#!coding=utf8
'''
read and write configuration in settings.cfg
'''
import sys
if sys.version_info.major == 3:
  from configparser import RawConfigParser as ConfigParser
else:
  from ConfigParser import SafeConfigParser as ConfigParser


####
class FileLikeString:
  '''helper class for preserving comments in config files'''
  def __init__(self):
    self.string = ""

  def write(self, chars):
    self.string += chars

  def splitlines(self):
    self.string = self.string.splitlines(True)

  def __getitem__(self, index):
    return self.string[index]

  def insert(self, index, value):
    self.string.insert(index, value)


def parse_value(string):
  # через try-except пишется проще, чем парсить минусы, дробные точки, 16-ричные числа и и т.п.
  # get integer
  try:
    return int(string)
  except ValueError:
    # or get binary
    try:
      return int(string, 2)
    except ValueError:
      # or get hexadicimal
      try:
        return int(string, 16)
      except ValueError:
        # or get float
        try:
          return float(string)
        except ValueError:
          # or string
          return string.strip("\"' ")
       


class DeviceConfig:
  # TODO: секция "WorkingSet" c номерами и именами рабочих точек
  def __init__(self, section):
    self.section = section
    self.config = ConfigParser()
    self.config.optionxform = str
    self.filename = "settings.cfg"
    self.config.read(self.filename)
    # read config and copy to the current namespace
    for i in self.config.options(self.section):
      value = self.config.get(self.section, i).strip()
      if "," in value: # array
        setattr(self, i, [])
        for j in value.strip("[ ]").split(","):
          getattr(self, i).append(parse_value(j.strip()) )
      else:            # single value
        setattr(self, i, parse_value(value))

      
####
  def save_config(self):
    """updates settings.cfg and preserves separate-line comments"""
    self.config.read(self.filename)
    for name in self.config.options(self.section):
      value = getattr(self, name)
      if name == "REG":
        self.config.set(self.section, name, hex(value))
      elif "ADDR" in name and "REG" in self.config.options(self.section) \
                          and getattr(self, name).__class__ == list:
        self.config.set(self.section, name, "[%d, 0x%x]" % (value[0], value[1]))
      else:
        self.config.set(self.section, name, str(value))
    fakefile = FileLikeString()
    self.config.write(fakefile)
    fakefile.splitlines()
    with open("settings.cfg","r") as f:
      oldfile = f.readlines()
    for num, old in enumerate(oldfile):
      if old.startswith("#") or old.startswith(";"):
        fakefile.insert(num, old)
    with open("settings.cfg","w") as newfile:
      newfile.writelines(fakefile)
#    print("save_config")


