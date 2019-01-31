// Generated by gencpp from file styx_msgs/TrafficLight.msg
// DO NOT EDIT!


#ifndef STYX_MSGS_MESSAGE_TRAFFICLIGHT_H
#define STYX_MSGS_MESSAGE_TRAFFICLIGHT_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <geometry_msgs/PoseStamped.h>

namespace styx_msgs
{
template <class ContainerAllocator>
struct TrafficLight_
{
  typedef TrafficLight_<ContainerAllocator> Type;

  TrafficLight_()
    : header()
    , pose()
    , state(0)  {
    }
  TrafficLight_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , pose(_alloc)
    , state(0)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef  ::geometry_msgs::PoseStamped_<ContainerAllocator>  _pose_type;
  _pose_type pose;

   typedef uint8_t _state_type;
  _state_type state;


    enum { UNKNOWN = 4u };
     enum { GREEN = 2u };
     enum { YELLOW = 1u };
     enum { RED = 0u };
 

  typedef boost::shared_ptr< ::styx_msgs::TrafficLight_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::styx_msgs::TrafficLight_<ContainerAllocator> const> ConstPtr;

}; // struct TrafficLight_

typedef ::styx_msgs::TrafficLight_<std::allocator<void> > TrafficLight;

typedef boost::shared_ptr< ::styx_msgs::TrafficLight > TrafficLightPtr;
typedef boost::shared_ptr< ::styx_msgs::TrafficLight const> TrafficLightConstPtr;

// constants requiring out of line definition

   

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::styx_msgs::TrafficLight_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::styx_msgs::TrafficLight_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace styx_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'std_msgs': ['/opt/ros/indigo/share/std_msgs/cmake/../msg'], 'sensor_msgs': ['/opt/ros/indigo/share/sensor_msgs/cmake/../msg'], 'geometry_msgs': ['/opt/ros/indigo/share/geometry_msgs/cmake/../msg'], 'styx_msgs': ['/home/rikki/repo/Self_Driving_Car/CarND-Capstone/ros/src/styx_msgs/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::styx_msgs::TrafficLight_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::styx_msgs::TrafficLight_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::styx_msgs::TrafficLight_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::styx_msgs::TrafficLight_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::styx_msgs::TrafficLight_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::styx_msgs::TrafficLight_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::styx_msgs::TrafficLight_<ContainerAllocator> >
{
  static const char* value()
  {
    return "444a7e648c334df4cc0678bcfbd971da";
  }

  static const char* value(const ::styx_msgs::TrafficLight_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x444a7e648c334df4ULL;
  static const uint64_t static_value2 = 0xcc0678bcfbd971daULL;
};

template<class ContainerAllocator>
struct DataType< ::styx_msgs::TrafficLight_<ContainerAllocator> >
{
  static const char* value()
  {
    return "styx_msgs/TrafficLight";
  }

  static const char* value(const ::styx_msgs::TrafficLight_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::styx_msgs::TrafficLight_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
geometry_msgs/PoseStamped pose\n\
uint8 state\n\
\n\
uint8 UNKNOWN=4\n\
uint8 GREEN=2\n\
uint8 YELLOW=1\n\
uint8 RED=0\n\
\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
\n\
================================================================================\n\
MSG: geometry_msgs/PoseStamped\n\
# A Pose with reference coordinate frame and timestamp\n\
Header header\n\
Pose pose\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Pose\n\
# A representation of pose in free space, composed of postion and orientation. \n\
Point position\n\
Quaternion orientation\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Point\n\
# This contains the position of a point in free space\n\
float64 x\n\
float64 y\n\
float64 z\n\
\n\
================================================================================\n\
MSG: geometry_msgs/Quaternion\n\
# This represents an orientation in free space in quaternion form.\n\
\n\
float64 x\n\
float64 y\n\
float64 z\n\
float64 w\n\
";
  }

  static const char* value(const ::styx_msgs::TrafficLight_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::styx_msgs::TrafficLight_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.pose);
      stream.next(m.state);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct TrafficLight_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::styx_msgs::TrafficLight_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::styx_msgs::TrafficLight_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "pose: ";
    s << std::endl;
    Printer< ::geometry_msgs::PoseStamped_<ContainerAllocator> >::stream(s, indent + "  ", v.pose);
    s << indent << "state: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.state);
  }
};

} // namespace message_operations
} // namespace ros

#endif // STYX_MSGS_MESSAGE_TRAFFICLIGHT_H
