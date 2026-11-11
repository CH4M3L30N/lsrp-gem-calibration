#!coding=utf-8
from motor import Motor
from device_config import DeviceConfig
import serial, time, threading
try:
  import epics
  use_epics = True
except ModuleNotFoundError:
  print("Working without EPICS. Auto mode can not be used.")
  use_epics = False
DEBUG = True

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Support(DeviceConfig):
  """Two-axis stepper-driver support"""
  def __init__(self, config_section, preset=None):
    '''Initialises the mirror.
    Checks power and status, defines preset if coordinates corresponds to any preset.
    <config_section> = "ElectronMirror" or "PositronMirror"
    '''
    super().__init__(config_section)
    self.dev0 = Motor(self.ADDR[0], self.PORT, self.BAUDRATE)  # motor # 1
    self.dev1 = Motor(self.ADDR[1], self.PORT, self.BAUDRATE)  # motor # 2
    self.status = {}
    if use_epics:
        self.pv_depol = epics.PV("lsrp:depolarizer:frequency")
        self.pv_x = epics.PV("lsrp:spot:x")
        self.pv_y = epics.PV("lsrp:spot:y")
        self.pv_dx = epics.PV("lsrp:spot:x_error")
        self.pv_dy = epics.PV("lsrp:spot:y_error")
        self.pv_sup_x = epics.PV("lsrp:gem_support:x")
        self.pv_sup_y = epics.PV("lsrp:gem_support:y")

    self.check_device_power()
    self.check_device_status()
    self.thread_sync = threading.Event()
    
    self.preset = None
    if DEBUG:
      self.dump_file = "support_position.dat"
      with open(self.dump_file, "w") as f:
        f.write("%.3f\t%.3f\n" % (self.status["x"]*self.SCALE[0],
        					      self.status["y"]*self.SCALE[1]))
    if preset is not None:
      self.preset = preset
    else:
      for p, x in enumerate(self.X):
        if self.status["x"] == x and self.status["y"] == self.Y[p]:
          self.preset = p
          break
    print(config_section, ": preset :", self.preset)



####
  def move(self, direction, step):
    '''Moves mirror motors at <direction> = {"x", "y"} at given amount of <steps>.'''
    self.check_device_power()
    '''Moves 90-degree pair of motors at <direction> = {"x", "y"} at given amount of <steps>.'''
    if step == 0: return
    if direction == "x":
      if not ((step*self.DIR[0] > 0 and self.dev0.status_byte["K+"]) or
              (step*self.DIR[0] < 0 and self.dev0.status_byte["K-"])):
        self.dev0.go_with_acc(self.DIR[0]*int(step))
        self.dev0.wait_until_ready()
      else: return
    elif direction=="y":
      if not ((step*self.DIR[1] > 0 and self.dev1.status_byte["K+"]) or
              (step*self.DIR[1] < 0 and self.dev1.status_byte["K-"])):
        self.dev1.go_with_acc(self.DIR[1]*int(step))
        self.dev1.wait_until_ready()
      else: return
    #-------waiting for readiness of both motors
    self.check_device_status()
    #-------if device met an edge: do nothing
    if DEBUG:
      with open(self.dump_file, "w") as f:
        f.write("%.3f\t%.3f\n" % (self.status["x"]*self.SCALE[0],
        					      self.status["y"]*self.SCALE[1]))


####
  def reset_coordinate(self):
    '''Sets coordinates of mirror motors to zero'''
    self.check_device_power()
    self.dev0.set_abs_position(0)
    self.dev1.set_abs_position(0)
    self.dev0.wait_until_ready()
    self.dev1.wait_until_ready()
    self.check_device_status()
    
    if DEBUG:
      with open(self.dump_file, "w") as f:
        f.write("%.3f\t%.3f\n" % (self.status["x"]*self.SCALE[0],
        					      self.status["y"]*self.SCALE[1]))


  def stop(self):
    self.check_device_power()
    self.dev0.immediate_stop()
    self.dev1.immediate_stop()
    self.dev0.wait_until_ready()
    self.dev1.wait_until_ready()
    self.check_device_status()
    

####
  def move_to_zero(self):
    '''Move mirrors to current 'central' position'''
    self.check_device_power()
    self.check_device_status()
    step0 = -self.dev0.position
    step1 = -self.dev1.position
    if step0 and step1:
      self.dev0.block()
      self.dev1.block()
    if step0: self.dev0.go_with_acc(step0)
    if step1: self.dev1.go_with_acc(step1)
    if step0 and step1: self.dev0.release()
    self.dev0.wait_until_ready()
    self.dev1.wait_until_ready()
    self.check_device_status()
    if DEBUG:
      with open(self.dump_file, "w") as f:

        f.write("%d\t%d\n" % (self.status["x"], self.status["y"]))

####
  def move_to(self, x, y=None, long_range=False):
    '''Move mirrors to any {x, y} position or preset'''
    if y is None: # x = preset number
      X = self.X[x]
      Y = self.Y[x]
      self.preset = x
    else:
      X = x
      Y = y
    self.check_device_power()
    self.check_device_status()
    step0 = int(X*self.DIR[0] - self.dev0.position) 
    step1 = int(Y*self.DIR[1] - self.dev1.position) 
    if step0 and step1 and (not long_range):
      self.dev0.block()
      self.dev1.block()
    if step0:
      self.dev0.go_with_acc(step0)
      self.dev0.wait_until_ready()
    if step1:
      self.dev1.go_with_acc(step1)
    if step0 and step1 and (not long_range):
      self.dev0.release()
    self.dev0.wait_until_ready()
    self.dev1.wait_until_ready()
    self.check_device_status()
    if DEBUG:
      with open(self.dump_file, "w") as f:
        f.write("%.3f\t%.3f\n" % (self.status["x"]*self.SCALE[0],
        					      self.status["y"]*self.SCALE[1]))

####
  def check_device_power(self):
    '''Checks if motors are connected'''
    for dev in (self.dev0, self.dev1):
      reply = self.dev0.get_device_info()
      if reply["Version"] == "0.0":
        self.status["state"] = "power fail"
        raise PowerFail(dev)

####
  def check_device_status(self):
    '''Check motors status and coordinates'''
    self.dev0.get_device_status()
    self.dev1.get_device_status()
    self.dev0.get_abs_position()
    self.dev1.get_abs_position()
    self.status["K+"] = [self.dev0.status_byte["K+"], self.dev1.status_byte["K+"]]
    self.status["K-"] = [self.dev0.status_byte["K-"], self.dev1.status_byte["K-"]]
    self.status["x"] = self.DIR[0]*self.dev0.position
    self.status["y"] = self.DIR[1]*self.dev1.position
    self.status["x_edges"] = [
               self.dev0.status_byte["K-"] if self.DIR[0] == 1 else self.dev0.status_byte["K+"],
               self.dev0.status_byte["K+"] if self.DIR[0] == 1 else self.dev0.status_byte["K-"]
                             ]
    self.status["y_edges"] = [
               self.dev1.status_byte["K-"] if self.DIR[1] == 1 else self.dev1.status_byte["K+"],
               self.dev1.status_byte["K+"] if self.DIR[1] == 1 else self.dev1.status_byte["K-"]]
    self.status["state"] = "ready" if self.dev0.status_byte["ready"] and self.dev1.status_byte["ready"] \
                 else "busy"
    if use_epics:
      self.pv_sup_x.put(self.SCALE[0]*self.status["x"], timeout=0.02)
      self.pv_sup_y.put(self.SCALE[1]*self.status["y"], timeout=0.02)


####
  def save_config(self, preset=None):
    '''Renewes preset and saves configuration in cfg file'''
    if preset is not None:
      self.preset = preset
    if self.preset is None:
      print("Preset was not saved")
      return
    if self.preset == len(self.X):
      # add new preset
      self.X.append(self.status["x"])
      self.Y.append(self.status["y"])
    else:
      self.X[self.preset] = self.status["x"]
      self.Y[self.preset] = self.status["y"]
    super().save_config()
  
####
  def delete_preset(self, preset):
    self.X.pop(preset)
    self.Y.pop(preset)
    if self.preset > preset:
      self.preset -= 1
    super().save_config()


######
  def calibrate(self):
    self.move_to(self.ZERO[0] - 1000*self.DIR[0], self.ZERO[1] - 1000*self.DIR[1], long_range=True)

    def seek(seek_range, slow=True):
      if slow:
        self.dev0.block()
        self.dev1.block()
        self.dev0.go_wo_acc(seek_range*self.K0_DIR[0], stop_cond=0b00100000)
        self.dev1.go_wo_acc(seek_range*self.K0_DIR[1], stop_cond=0b00100000)
        self.dev0.release()
        self.dev1.release()
      else:
        self.dev0.go_with_acc(seek_range*self.K0_DIR[0])
        self.dev0.wait_until_ready()
        self.dev1.go_with_acc(seek_range*self.K0_DIR[1])
        self.dev1.wait_until_ready()
      self.dev0.get_device_status()
      self.dev1.get_device_status()
 # первое сканирование
    seek(2000)
    
    if (not self.dev0.status_byte["K0"]) or (not self.dev1.status_byte["K0"]):
      # идти в направлении K0 до K+
      seek(60000, False)
      # после концевика отойти немного
      seek(-1000, False)
      # сканировать
      seek(2000)
      for dev in (self.dev0, self.dev1):
        if not dev.status_byte["K0"]:
          print("Seems that K0 of the motor #{} (S/N {}) doesn't work!".format(dev.N,
                                                                        dev.get_device_info()["SN"]))
    self.check_device_status()
    print("Old and new calibration difference: dx={}, dy={}.".format(
          self.status["x"] - self.ZERO[0]*self.DIR[0], self.status["y"] - self.ZERO[1]*self.DIR[1]))
    self.dev0.set_abs_position(self.ZERO[0]*self.DIR[0])
    self.dev1.set_abs_position(self.ZERO[1]*self.DIR[1])
    self.check_device_status()



  def wait_tk(self, exit_message="Exit", timeout=20):
    w = self.thread_sync.wait(timeout)
    self.thread_sync.clear()
    if not w:
      raise threading.ThreadError


###################################
  def auto(self):
    print("Auto was started.")
    try:
      self.status["state"] = "Auto"
      spot = True
      depol_works = True
      while True:
#        dx_spot, dy_spot = 0, 0
        try:
          self.wait_tk(self.REFR_TIME*2.1)
          x = self.pv_x.get(timeout=0.01)
          y = self.pv_y.get(timeout=0.01)
          dx = self.pv_dx.get(timeout=0.01)
          dy = self.pv_dx.get(timeout=0.01)
          
          depol_freq = self.pv_depol.get(timeout=0.01)
          if not (depol_freq is None) and depol_freq > 100.:
            if depol_works:
              print("Depolarizer works.")
              depol_works = False
            time.sleep(self.REFR_TIME)
            continue
          else:
            depol_works = True
          
          self.pv_x.get_timevars()
          if (None in [x, y]) or (None in [dx, dy]) or \
          			time.time() - self.pv_x.timestamp > 2.*self.REFR_TIME:
            if spot:
              print("No spot, preprocess or connection error.")
              spot = False
            time.sleep(self.REFR_TIME)
            continue
          else:
            spot = True
          x_spot = x / self.SCALE[0]
          y_spot = y / self.SCALE[1]
          dx_spot = dx / self.SCALE[0]
          dy_spot = dy / self.SCALE[1]
          print("Spot: X = {:.3f}+-{:.3f}, \tY = {:.3f}+-{:.3f}".format(x_spot*self.SCALE[0],
                        dx_spot*self.SCALE[0], y_spot*self.SCALE[1], dy_spot*self.SCALE[1]))
          if ((abs(x_spot) > self.AUTO_THR and abs(x_spot/dx_spot) > 3.)  or
              (abs(y_spot) > self.AUTO_THR and abs(y_spot/dy_spot) > 3.)) and \
             (abs(x_spot) < 25./self.SCALE[0]  or abs(y_spot) < 10./self.SCALE[1]):
            print("Correcting position.")
            self.move_to(self.status["x"] - x_spot, self.status["y"] - y_spot)
#            time.sleep(2*self.REFR_TIME)
        except epics.ca.ChannelAccessGetFailure:
          print("EPICS `get` error!")
#        except:
#          print("\tNetwork error!")
        for i in range(2):
          self.wait_tk(self.REFR_TIME*1.5)
          time.sleep(self.REFR_TIME)
    except PowerFail:
      print("Motor power fail!")
    except serial.SerialException:
      print("Motor connection fail!")
    except threading.ThreadError:
      print("Auto was stopped.")
  

# exception classes
class PowerFail(Exception):
  def __init__(self, dev):
    print("No power applied at device", dev, "!")

