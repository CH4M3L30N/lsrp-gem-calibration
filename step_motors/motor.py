from __future__ import print_function
import serial, select, os
import sys
if sys.version_info.major == 3:

  def serial_decode(pin):
    return pin.decode("latin-1")

  def serial_encode(pin):
    return bytes(pin, "latin-1")

else:

  def serial_decode(pin):
    return pin
  
  def serial_encode(pin):
    return pin

class Motor:
  status_byte = {
         "BC" : 0,
         "hitK" : 0, 
         "accu" : 0, 
         "K0"   : 0, 
         "K+"   : 0, 
         "K-"   : 0, 
         "move" : 0, 
         "ready": 0}
         
  def __init__(self, addr=1, ioport="/dev/ttyUSB0", baudrate=19200, debug=False, timeout=0.1):
    self.N, self.ioPORT = int(addr) & 0xFF, ioport
    self.LockFile = "/tmp/" + os.path.split(ioport)[1] + "_lock"
    if not os.path.exists(self.LockFile):
      os.mkfifo(self.LockFile)
    self.debug, self.timeout = debug, timeout
    self.IORates = {1200: 0, 2400: 1, 4800: 2, 9600:3, 19200:4, 38400:5, 57600:6}
    if baudrate not in self.IORates.keys():
      raise IOError("Non-standard baud rate, must be 1200, 2400, 4800, 9600, 19200, 38400, or 57600.")
    self.BaudRate = baudrate
    self.START, self.STOP, self.SHIFT = chr(0xAA), chr(0xAB), chr(0xAC)

  def get_config_byte(self, cfg):
    """Get configuration byte. <cfg> is dictionary."""
    B0 = 0 if "4" in cfg["Mode"]    else 1
    B2 = 0 if "Closed" in cfg["K-"] else 1
    B3 = 0 if "Closed" in cfg["K+"] else 1
    B4 = 0 if "Closed" in cfg["K0"] else 1
    B5 = int(cfg["SoftTrailers"])
    B6 = int(cfg["LeaveTrailer"])
    B7 = int(cfg["AccOnLeave"])
    return B0 | (B2 << 1) | (B3 << 2) | (B4 << 3) | (B5 << 4) | (B6 << 5) | (B7 << 6)

  def get_status_byte(self, reply):
    """Get status byte from the device reply. Returns <self.status_byte>"""
    self.status_byte = {"BC"    : (reply[1] >> 7) & 1,
                       "hitK"  : (reply[1] >> 6) & 1,
                       "accu"  : (reply[1] >> 5) & 1,
                       "K0"    : (reply[1] >> 4) & 1,
                       "K+"    : (reply[1] >> 3) & 1,
                       "K-"    : (reply[1] >> 2) & 1,
                       "move"  : (reply[1] >> 1) & 1,
                       "ready" :  reply[1] & 1}
    return self.status_byte

# Device will answer to the commands, marked with "*" even if it is busy

  def set_logical_address(self, sn):  # ...................................................Command "0"
    """Command 0: set logical address <sn>."""
    packet = chr(0) + self.START + "WS" + chr((sn & 0xFF00) >> 8) + chr(sn & 0xFF) + chr(self.IORates[self.BaudRate]) + chr(self.N)
    ser = serial.Serial(self.ioPORT, 600, timeout=5)
    ser.flushInput()
    ser.flushOutput()
    ser.write(serial_encode(packet))
    ser.close()


  def get_device_info(self):  # .........................................................Command 1 *
    """Command 1: get the device info as a dictionary"""
    reply = self.__exchange__([1], 8)
    return {"Version": str(reply[3]) + "." + str(reply[4]), "SN": (reply[5] << 8) + reply[6] }

  def repeat_last_answer(self, reply_length):  # ......................................................Command 2 *
    """Command 2: get the device info as a dictionary"""
    self.__exchange__([2], reply_length)
    
  def get_device_status(self):  # .........................................................Command 3 *
    reply = self.__exchange__([3])
    return self.get_status_byte(reply)

  def go_with_acc(self, steps, stop_cond=0):  # ..............................................Command 4
    N = int(steps)
    self.wait_until_ready()
    reply = self.__exchange__([4, (N >> 24) & 0xFF, (N >> 16) & 0xFF, (N >> 8) & 0xFF, N & 0xFF, stop_cond & 0xFF])
    self.get_status_byte(reply)

  def go_wo_acc(self, steps, stop_cond=0): # ................................................Command 5
    N = int(steps)
    self.wait_until_ready()
    reply = self.__exchange__([5, (N >> 24) & 0xFF, (N >> 16) & 0xFF, (N >> 8) & 0xFF, N & 0xFF, stop_cond & 0xFF])
    self.get_status_byte(reply)

  def set_device_config(self, i_move, i_stop, t_stop, cfg):  # ...................................Command 6
    i_move, i_stop, t_stop, cfg = int(i_move) & 0xFF, int(i_stop) & 0xFF, int(t_stop) & 0xFF, int(cfg) & 0xFF
    self.wait_until_ready()
    reply = self.__exchange__([6, i_move, i_stop, t_stop, cfg])
    self.get_status_byte(reply)
  
  def set_device_speed(self, v_min, v_max, acc): # ............................................Command 7
    self.wait_until_ready()
    reply = self.__exchange__([7, (v_min >> 8) & 0xFF, v_min & 0xFF, (v_max >> 8) & 0xFF,
                               v_max & 0xFF, (acc >> 8) & 0xFF, acc & 0xFF])
    self.get_status_byte(reply)
  
  def immediate_stop(self): #.............................................................Command 8 *
    reply = self.__exchange__([8])
    self.get_status_byte(reply)
  
  def current_off(self): #................................................................Command 9
    self.wait_until_ready()
    reply = self.__exchange__([9])
    self.get_status_byte(reply)
  
  def save_data(self): #..................................................................Command 10
    self.wait_until_ready()
    reply = self.__exchange__([10])
    self.get_status_byte(reply)
  
  def blink_on_sync(self, n=0, start=1, step=1): #...........................................Command 11
    self.wait_until_ready()
    reply = self.__exchange__([11, (n >> 8) & 0xFF, n & 0xFF, (start >> 8) & 0xFF,
                               start & 0xFF, (step >> 8) & 0xFF, step & 0xFF])
    self.get_status_byte(reply)
  
  def get_rest_steps(self): #.............................................................Command 12
    self.wait_until_ready()
    reply = self.__exchange__([12], 6)
    return int(reply[1] << 24 | reply[2] << 16 | reply[3]<<8 | reply[4] -
                              ((reply[1] & 0x80) << 25))
  
  def get_device_config(self): # .........................................................Command 13
    self.wait_until_ready()
    reply = self.__exchange__([13], 6)
    return {"IMove": reply[1], "IStop": reply[2], "TStop": reply[3], "CFG": reply[4]}
  
  def get_device_speed(self): # ..........................................................Command 14
    self.wait_until_ready()
    reply = self.__exchange__([14], 8)
    return {"Vmin": (reply[1] << 8) + reply[2], "Vmax": (reply[3] << 8) + reply[4], "Acc": (reply[5] << 8) + reply[6]}

  def frequency_test(self): #.............................................................Command 15
    self.wait_until_ready()
    self.__exchange__([15], 0)

  def set_period(self, t): #...............................................................Command 16
    self.wait_until_ready()
    reply = self.__exchange__([16, (t >> 24) & 0xFF, (t >> 16) & 0xFF, (t >> 8) & 0xFF, t & 0xFF])
    self.get_status_byte(reply)

  def go_accurately(self, steps, time): # ..................................................Command 17
    s, t = int(steps), int(time)
    self.wait_until_ready()
    reply = self.__exchange__([17, (s >> 24) & 0xFF, (s >> 16) & 0xFF, (s >> 8) & 0xFF, s & 0xFF,
                                   (t >> 24) & 0xFF, (t >> 16) & 0xFF, (t >> 8) & 0xFF, t & 0xFF])
    self.get_status_byte(reply)

  def give_pulse(self): #.................................................................Command 18
    self.wait_until_ready()
    reply = self.__exchange__([18])
    self.get_status_byte(reply)

  def set_abs_position(self, k): #.........................................................Command 19
    K = int(k)
    self.wait_until_ready()
    reply = self.__exchange__([19, (K >> 24) & 0xFF, (K >> 16) & 0xFF, (K >> 8) & 0xFF, K & 0xFF])
    self.get_status_byte(reply)

  def get_abs_position(self): #...........................................................Command 20
    self.wait_until_ready()
    reply = self.__exchange__([20], 6)
    self.position = int(reply[1]<<24 | reply[2]<<16 | reply[3]<<8 | reply[4] - ((reply[1] & 0x80) << 25))

  def block(self): #......................................................................Command 21
    self.wait_until_ready()
    reply = self.__exchange__([21, 3]) # Block movement, enable broadcast
    self.get_status_byte(reply)
    
  def release(self): #....................................................................Command 21
    tmp = self.N
    self.wait_until_ready() # No answer if N=0 !!!
    self.N = 0
    self.__exchange__([21, 4]) # UnBlock movement, disable broadcast
    self.N = tmp
    
  def write_user_data(self, data): #.......................................................Command 22
    s = str(data)
    for i in range(16):
      d = list(s[i*8: (i+1)*8])
      if len(d) > 0:
        while len(d)<8: d.extend(" ")
        print(d)
        self.wait_until_ready()
        reply = self.__exchange__([22, i, ord(d[0]), ord(d[1]), ord(d[2]), ord(d[3]), ord(d[4]), ord(d[5]),
                                          ord(d[6]), ord(d[7])])
        self.get_status_byte(reply)
      else:
        break
    
  def read_user_data(self): #.............................................................Command 23
    s = ""
    for i in range(16):
      self.wait_until_ready()
      reply = self.__exchange__([23, i], 10)
      for j in range(1, 9):
        s += chr(reply[j])
    return s

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  def wait_until_ready(self):
    self.get_device_status()
    while not self.status_byte["ready"]:
      self.get_device_status()

  def __exchange__(self, body, reply_length=3): # ....................The Main Packets Exchange Algorythm
# 1. Preparation of outgouing packet:
    out, xor = self.START + chr(self.N), self.N
    for byte in body:
      xor ^= byte
      if chr(byte) in [self.START, self.STOP, self.SHIFT]:
        out += self.SHIFT + chr(byte - ord(self.START))
      else:
        out += chr(byte)
    if chr(xor) in [self.START, self.STOP, self.SHIFT]:
      out += self.SHIFT + chr(xor - (ord(self.START)))
    else:
      out += chr(xor)
    out += self.STOP
# 2. Exchange:
    if self.debug: self.show_string("OUT", out)
    try:
      lock = os.open(self.LockFile, os.O_RDWR)
      while True:
        if not select.select([lock],[lock],[])[0]: break # empty pipe
      os.write(lock, b"1")
      ser = serial.Serial(self.ioPORT, self.BaudRate, timeout=self.timeout)
      ser.flushInput()
      ser.flushOutput()
      ser.write(serial_encode(out))
#      pin = serial_decode(ser.read_until(self.STOP))
      pin = serial_decode(ser.read(12))
      ser.close()
      os.read(lock, 1)
      os.close(lock)
    except:
      os.close(lock)
      raise
    
    if self.debug:  self.show_string("REPLY", pin)
# 3. Translating the Reply:
    reply = []
    if len(pin) > 1:
      shift_in_reply, xor = 0, 0
      pin = pin[0:len(pin)-1]
      for c in pin:
#        c = chr(c)
        if c == self.SHIFT:
          shift_in_reply = ord(self.START)
        else:
          reply.append(ord(c) + shift_in_reply)
          xor ^= ord(c) + shift_in_reply
          shift_in_reply = 0
# 4. Check possible errors, if any:
      if xor:
        print("Wrong checksum at device #%d!" % self.N)
        self.repeat_last_answer(reply_length)
#      if reply[0]<<8 + reply[0] != self.N:
      if reply[0] != self.N:
        print("Wrong packet at device #%d!" % self.N)
        self.repeat_last_answer(reply_length)
      if len(reply) != reply_length:
        print("Wrong packet length at device #%d" % self.N)
        self.repeat_last_answer(reply_length)
      return reply
    else:
      return [0 for elem in range(reply_length)]

  def show_string(self, title, s):
    print(title + ":")
    for c in s:
#      print("Dec: %03d | Hex:%2X | Str:%1c" % (ord(c), ord(c), c))
      print("Dec: %03d | Hex:%2X" % (ord(c), ord(c)))

