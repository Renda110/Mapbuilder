


#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>


#include <boost/circular_buffer.hpp>
#include <boost/optional.hpp>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <boost/assign.hpp>

using std::vector;
using std::list;
using std::string;
using boost::assign::operator+=;
//using namespace boost::assign;
typedef boost::shared_ptr<nav_msgs::OccupancyGrid> GridPtr;
typedef boost::shared_ptr<nav_msgs::OccupancyGrid const> GridConstPtr;
list<GridPtr> grids[MAX_ROBOTS];
list<geometry_msgs::PoseWithCovarianceStamped> robot_pose[MAX_ROBOTS];
//calum
namespace map_store_node
{

class Map_Store_Node
{
public:
  Map_Store_Node ();

  ros::Publisher map_pub_;

private:
  void robot1_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid);
  void robot1_poseCallback (const geometry_msgs::PoseWithCovarianceStamped& pose);
  void robot2_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid);
  void robot2_poseCallback (const geometry_msgs::PoseWithCovarianceStamped& pose);
  void robot3_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid);
  void robot3_poseCallback (const geometry_msgs::PoseWithCovarianceStamped& pose);
  void robot4_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid);
  void robot4_poseCallback (const geometry_msgs::PoseWithCovarianceStamped& pose);
  void robot5_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid);
  void robot5_poseCallback (const geometry_msgs::PoseWithCovarianceStamped& pose);

    ros::Subscriber robot1_grid_sub_;
    ros::Subscriber robot2_grid_sub_;
    ros::Subscriber robot3_grid_sub_;
    ros::Subscriber robot4_grid_sub_;
    ros::Subscriber robot5_grid_sub_;
    ros::Subscriber robot1_pose_sub_;
    ros::Subscriber robot2_pose_sub_;
    ros::Subscriber robot3_pose_sub_;
    ros::Subscriber robot4_pose_sub_;
    ros::Subscriber robot5_pose_sub_;



};

Map_Store_Node::Map_Store_Node ()
{
    ros::NodeHandle nh("");
    robot1_grid_sub_=nh.subscribe("/wambot1/submap", 1, &Map_Store_Node::robot1_gridCallback, this);
    robot1_pose_sub_=nh.subscribe("/wambot1/robot_pose", 1, &Map_Store_Node::robot1_poseCallback, this);
    robot2_grid_sub_=nh.subscribe("/wambot2/submap", 1, &Map_Store_Node::robot2_gridCallback, this);
    robot2_pose_sub_=nh.subscribe("/wambot2/robot_pose", 1, &Map_Store_Node::robot2_poseCallback, this);
    robot3_grid_sub_=nh.subscribe("/wambot3/submap", 1, &Map_Store_Node::robot3_gridCallback, this);
    robot3_pose_sub_=nh.subscribe("/wambot3/robot_pose", 1, &Map_Store_Node::robot3_poseCallback, this);
    robot4_grid_sub_=nh.subscribe("/wambot4/submap", 1, &Map_Store_Node::robot4_gridCallback, this);
    robot4_pose_sub_=nh.subscribe("/wambot4/robot_pose", 1, &Map_Store_Node::robot4_poseCallback, this);
    robot5_grid_sub_=nh.subscribe("/wambot5/submap", 1, &Map_Store_Node::robot5_gridCallback, this);
    robot5_pose_sub_=nh.subscribe("/wambot5/robot_pose", 1, &Map_Store_Node::robot5_poseCallback, this);
    map_pub_ = nh.advertise<nav_msgs::OccupancyGrid>("map", 1);
}

void Map_Store_Node::robot1_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid){
    grids[1] += grid;
    ROS_INFO("Received robot1 grid (buffer %d)",(uint)grids[1].size());
}
void Map_Store_Node::robot2_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid){
    grids[2] += grid;
    ROS_INFO("Received robot2 grid (buffer %d)",(uint)grids[2].size());
}
void Map_Store_Node::robot3_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid){
    grids[3] += grid;
    ROS_INFO("Received robot3 grid (buffer %d)",(uint)grids[3].size());
}
void Map_Store_Node::robot4_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid){
    grids[4] += grid;
    ROS_INFO("Received robot4 grid (buffer %d)",(uint)grids[4].size());
}
void Map_Store_Node::robot5_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid){
    grids[5] += grid;
    ROS_INFO("Received robot5 grid (buffer %d)",(uint)grids[5].size());
}

void Map_Store_Node::robot1_poseCallback (const geometry_msgs::PoseWithCovarianceStamped & pose){
    ROS_INFO("Received robot1 pose=%1.2f,%1.2f",pose.pose.pose.position.x,pose.pose.pose.position.y);
    robot_pose[1] += pose;
}
void Map_Store_Node::robot2_poseCallback (const geometry_msgs::PoseWithCovarianceStamped & pose){
    ROS_INFO("Received robot2 pose=%1.2f,%1.2f",pose.pose.pose.position.x,pose.pose.pose.position.y);
    robot_pose[2] += pose;
}
void Map_Store_Node::robot3_poseCallback (const geometry_msgs::PoseWithCovarianceStamped & pose){
    ROS_INFO("Received robot3 pose=%1.2f,%1.2f",pose.pose.pose.position.x,pose.pose.pose.position.y);
    robot_pose[3] += pose;
}
void Map_Store_Node::robot4_poseCallback (const geometry_msgs::PoseWithCovarianceStamped & pose){
    ROS_INFO("Received robot4 pose=%1.2f,%1.2f",pose.pose.pose.position.x,pose.pose.pose.position.y);
    robot_pose[4] += pose;
}
void Map_Store_Node::robot5_poseCallback (const geometry_msgs::PoseWithCovarianceStamped & pose){
    ROS_INFO("Received robot5 pose=%1.2f,%1.2f",pose.pose.pose.position.x,pose.pose.pose.position.y);
    robot_pose[5] += pose;
}


} // namespace map_store_node


//calum

