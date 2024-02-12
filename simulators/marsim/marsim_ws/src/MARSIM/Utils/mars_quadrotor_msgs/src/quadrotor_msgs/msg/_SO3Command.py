"""autogenerated by genpy from quadrotor_msgs/SO3Command.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import geometry_msgs.msg
import quadrotor_msgs.msg
import std_msgs.msg

class SO3Command(genpy.Message):
  _md5sum = "a466650b2633e768513aa3bf62383c86"
  _type = "quadrotor_msgs/SO3Command"
  _has_header = True #flag to mark the presence of a Header object
  _full_text = """Header header
geometry_msgs/Vector3 force
geometry_msgs/Quaternion orientation
float64[3] kR
float64[3] kOm
quadrotor_msgs/AuxCommand aux

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data
# in a particular coordinate frame.
#
# sequence ID: consecutively increasing ID
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
# 0: no frame
# 1: global frame
string frame_id

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space.

float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w

================================================================================
MSG: quadrotor_msgs/AuxCommand
float64 current_yaw
float64 kf_correction
float64[2] angle_corrections# Trims for roll, pitch
bool enable_motors
bool use_external_yaw

"""
  __slots__ = ['header','force','orientation','kR','kOm','aux']
  _slot_types = ['std_msgs/Header','geometry_msgs/Vector3','geometry_msgs/Quaternion','float64[3]','float64[3]','quadrotor_msgs/AuxCommand']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       header,force,orientation,kR,kOm,aux

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(SO3Command, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.force is None:
        self.force = geometry_msgs.msg.Vector3()
      if self.orientation is None:
        self.orientation = geometry_msgs.msg.Quaternion()
      if self.kR is None:
        self.kR = [0.,0.,0.]
      if self.kOm is None:
        self.kOm = [0.,0.,0.]
      if self.aux is None:
        self.aux = quadrotor_msgs.msg.AuxCommand()
    else:
      self.header = std_msgs.msg.Header()
      self.force = geometry_msgs.msg.Vector3()
      self.orientation = geometry_msgs.msg.Quaternion()
      self.kR = [0.,0.,0.]
      self.kOm = [0.,0.,0.]
      self.aux = quadrotor_msgs.msg.AuxCommand()

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_struct_3I.pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_struct_7d.pack(_x.force.x, _x.force.y, _x.force.z, _x.orientation.x, _x.orientation.y, _x.orientation.z, _x.orientation.w))
      buff.write(_struct_3d.pack(*self.kR))
      buff.write(_struct_3d.pack(*self.kOm))
      _x = self
      buff.write(_struct_2d.pack(_x.aux.current_yaw, _x.aux.kf_correction))
      buff.write(_struct_2d.pack(*self.aux.angle_corrections))
      _x = self
      buff.write(_struct_2B.pack(_x.aux.enable_motors, _x.aux.use_external_yaw))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(_x))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(_x))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.force is None:
        self.force = geometry_msgs.msg.Vector3()
      if self.orientation is None:
        self.orientation = geometry_msgs.msg.Quaternion()
      if self.aux is None:
        self.aux = quadrotor_msgs.msg.AuxCommand()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _struct_3I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 56
      (_x.force.x, _x.force.y, _x.force.z, _x.orientation.x, _x.orientation.y, _x.orientation.z, _x.orientation.w,) = _struct_7d.unpack(str[start:end])
      start = end
      end += 24
      self.kR = _struct_3d.unpack(str[start:end])
      start = end
      end += 24
      self.kOm = _struct_3d.unpack(str[start:end])
      _x = self
      start = end
      end += 16
      (_x.aux.current_yaw, _x.aux.kf_correction,) = _struct_2d.unpack(str[start:end])
      start = end
      end += 16
      self.aux.angle_corrections = _struct_2d.unpack(str[start:end])
      _x = self
      start = end
      end += 2
      (_x.aux.enable_motors, _x.aux.use_external_yaw,) = _struct_2B.unpack(str[start:end])
      self.aux.enable_motors = bool(self.aux.enable_motors)
      self.aux.use_external_yaw = bool(self.aux.use_external_yaw)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_struct_3I.pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      if python3:
        buff.write(struct.pack('<I%sB'%length, length, *_x))
      else:
        buff.write(struct.pack('<I%ss'%length, length, _x))
      _x = self
      buff.write(_struct_7d.pack(_x.force.x, _x.force.y, _x.force.z, _x.orientation.x, _x.orientation.y, _x.orientation.z, _x.orientation.w))
      buff.write(self.kR.tostring())
      buff.write(self.kOm.tostring())
      _x = self
      buff.write(_struct_2d.pack(_x.aux.current_yaw, _x.aux.kf_correction))
      buff.write(self.aux.angle_corrections.tostring())
      _x = self
      buff.write(_struct_2B.pack(_x.aux.enable_motors, _x.aux.use_external_yaw))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(_x))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(_x))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.force is None:
        self.force = geometry_msgs.msg.Vector3()
      if self.orientation is None:
        self.orientation = geometry_msgs.msg.Quaternion()
      if self.aux is None:
        self.aux = quadrotor_msgs.msg.AuxCommand()
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _struct_3I.unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8')
      else:
        self.header.frame_id = str[start:end]
      _x = self
      start = end
      end += 56
      (_x.force.x, _x.force.y, _x.force.z, _x.orientation.x, _x.orientation.y, _x.orientation.z, _x.orientation.w,) = _struct_7d.unpack(str[start:end])
      start = end
      end += 24
      self.kR = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=3)
      start = end
      end += 24
      self.kOm = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=3)
      _x = self
      start = end
      end += 16
      (_x.aux.current_yaw, _x.aux.kf_correction,) = _struct_2d.unpack(str[start:end])
      start = end
      end += 16
      self.aux.angle_corrections = numpy.frombuffer(str[start:end], dtype=numpy.float64, count=2)
      _x = self
      start = end
      end += 2
      (_x.aux.enable_motors, _x.aux.use_external_yaw,) = _struct_2B.unpack(str[start:end])
      self.aux.enable_motors = bool(self.aux.enable_motors)
      self.aux.use_external_yaw = bool(self.aux.use_external_yaw)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
_struct_2d = struct.Struct("<2d")
_struct_3I = struct.Struct("<3I")
_struct_7d = struct.Struct("<7d")
_struct_2B = struct.Struct("<2B")
_struct_3d = struct.Struct("<3d")