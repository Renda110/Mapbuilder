#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <map>
#include <set>
#include <list>

#include <mrpt/base/include/mrpt/poses/CPoseOrPoint.h>
#include <mrpt/base/include/mrpt/poses/CPose2D.h>

#include "MapBuilder.h"

#include "simplelog.h"
#include <time.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glx.h>

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <boost/circular_buffer.hpp>
#include <boost/optional.hpp>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <boost/assign.hpp>

char *CONFIG_FILE_NAME = (char*)"mapping.ini";

using namespace mrpt;
using namespace mrpt::synch;
using namespace mapping;
using namespace std;

using std::vector;
using std::list;
using std::string;
using boost::assign::operator+=;
//using namespace boost::assign;

typedef boost::shared_ptr<nav_msgs::OccupancyGrid> GridPtr;
typedef boost::shared_ptr<nav_msgs::OccupancyGrid const> GridConstPtr;

list<GridPtr> grids[MAX_ROBOTS];
list<geometry_msgs::PoseWithCovarianceStamped> robot_pose[MAX_ROBOTS];

geometry_msgs::PoseStamped robot5_map_pose;
tf::Transform tf_map_to_robot;

UUID robot5_current_uuid;
UUID robot4_current_uuid;
UUID robot3_current_uuid;
UUID robot2_current_uuid;
UUID robot1_current_uuid;
UUID *current_uuid[MAX_ROBOTS];

//check if we're the GCS or robot:
bool is_GCS       = true;
bool is_WAMBOT    = false;

//on one HMI pc only: environment variable
bool is_GCS_OPTIMISER = true;
bool is_GCS_MATCHER   = true;

Display *display;
Window window;
GLXContext glc;

gui_state_types gui_state = NONE;
Submap* submap_gui_target = 0;
double gui_rotating_phi_start = 0;
double dist_squared_best = 1e99;

//mouse callback
void mouse_callback( int event, int sx, int sy, int flags, void* param )
{
    MapBuilder *mapBuilder = (MapBuilder*)param;
    double map_mouse_x =  (sx / 20.0) + mapBuilder->x_min;
    double map_mouse_y = (-sy / 20.0) + mapBuilder->y_max; //20 corresponds to the CELL_PER_METER define in Mapbuilder.h

    if (gui_state == NONE && ( event == CV_EVENT_LBUTTONDOWN || event == CV_EVENT_MBUTTONDOWN || event == CV_EVENT_RBUTTONDOWN) )
    {
        //start gui action
        dist_squared_best = 1e99;

        int target_x_best=0; int target_y_best=0;
        submap_gui_target = 0;

        BOOST_FOREACH(Submap* submap, mapBuilder->map->GetSubmapList() )
        {
            if (submap->GetPose() && submap->GetOccupancyData() )
            {
                //check for closest submap origin
                CPose2D p = submap->GetPose()->GetMean();
                double dist_squared = p.distance2DToSquare( map_mouse_x, map_mouse_y );

                SIMPLELOGGER_DEBUG_MORE("dist=%1.2f x=%1.1f y=%1.1f ",sqrt(dist_squared),map_mouse_x,map_mouse_y);

                if (dist_squared<dist_squared_best)
                {
                    //close submap
                    submap_gui_target = submap;
                    dist_squared_best = dist_squared;

                    if (submap->GetGroundTruth())
                    {
                        p = submap->GetGroundTruth()->mean;
                    }
                    target_x_best = (p.x()-mapBuilder->x_min)* 20.0;
                    target_y_best = (p.y()-mapBuilder->y_max)*-20.0;
                }

                //check for closest ground truth node
                if (submap->GetGroundTruth())
                {
                    CPose2D p = submap->GetGroundTruth()->mean;
                    double dist_squared = p.distance2DToSquare( map_mouse_x, map_mouse_y );

                    SIMPLELOGGER_DEBUG("ground truth dist=%1.2f x=%1.1f y=%1.1f ",sqrt(dist_squared),map_mouse_x,map_mouse_y);

                    if (dist_squared < sqr(1))
                    {
                        dist_squared/=5;//if near a GT, prefer it
                    }

                    if (dist_squared<dist_squared_best)
                    {
                        submap_gui_target = submap;
                        dist_squared_best = dist_squared;
                        target_x_best = (p.x()-mapBuilder->x_min)* 20.0;
                        target_y_best = (p.y()-mapBuilder->y_max)*-20.0;
                    }
                }
            }
        }

        if (submap_gui_target &&(dist_squared_best<sqr(4.0)))
        {
            SIMPLELOGGER_INFO("CV_EVENT_xBUTTONDOWN on submap %s x=%1.1f y=%1.1f (%d,%d)",submap_gui_target->GetUUID().c_str(),map_mouse_x,map_mouse_y,sx,sy );

            //snap cursor to node
            XWarpPointer(display, None, window, 0, 0, 0, 0, target_x_best, target_y_best);
            XFlush(display);

            switch( event )
            {
                case CV_EVENT_LBUTTONDOWN:
                    SIMPLELOGGER_INFO("moving ground truth for submap %s x=%1.1f y=%1.1f (%d,%d)",submap_gui_target->GetUUID().c_str(),map_mouse_x,map_mouse_y,sx,sy );
                    gui_state = MOVING_CONSTRAINT;
                    break;
                case CV_EVENT_RBUTTONDOWN:
                    SIMPLELOGGER_INFO("rotating ground truth for submap %s x=%1.1f y=%1.1f (%d,%d)",submap_gui_target->GetUUID().c_str(),map_mouse_x,map_mouse_y,sx,sy );
                    gui_state = ROTATING_CONSTRAINT;
                    gui_rotating_phi_start = (submap_gui_target->GetGroundTruth()) ? submap_gui_target->GetGroundTruth()->mean.phi() : submap_gui_target->GetPose()->GetMean().phi();
                    break;
                case CV_EVENT_MBUTTONDOWN:
                    SIMPLELOGGER_INFO("deleting ground truth for submap %s x=%1.1f y=%1.1f (%d,%d)",submap_gui_target->GetUUID().c_str(),map_mouse_x,map_mouse_y,sx,sy );
                    mapBuilder->map->DeleteSubmapGroundTruth( submap_gui_target->GetUUID() );
                    break;
            }
        }
    }

    if (gui_state==MOVING_CONSTRAINT && event==CV_EVENT_RBUTTONDOWN )
    {
        gui_state = ROTATING_CONSTRAINT;
        gui_rotating_phi_start = (submap_gui_target->GetGroundTruth()) ? submap_gui_target->GetGroundTruth()->mean.phi() : submap_gui_target->GetPose()->GetMean().phi();
    }

    if (gui_state==ROTATING_CONSTRAINT && event==CV_EVENT_LBUTTONDOWN )
    {
        gui_state = MOVING_CONSTRAINT;
        //jump cursor to gt
        CPose2D p = submap_gui_target->GetPose()->GetMean();
        if (submap_gui_target->GetGroundTruth()) p = submap_gui_target->GetGroundTruth()->mean;
        int sx = (p.x()-mapBuilder->x_min)* 20.0;
        int sy = (p.y()-mapBuilder->y_max)*-20.0;
        XWarpPointer(display, None, window, 0, 0, 0, 0, sx, sy);
        XFlush(display);
    }

    if (gui_state!=NONE && (event==CV_EVENT_MOUSEMOVE || event==CV_EVENT_LBUTTONDOWN ||event==CV_EVENT_RBUTTONDOWN  ) )
    {
        SIMPLELOGGER_DEBUG("CV_EVENT_MOUSE x %d y %d",sx,sy);
        CPosePDFGaussian ground_truth;

        if (submap_gui_target->GetGroundTruth())
        { //already have one
            ground_truth.mean = submap_gui_target->GetGroundTruth()->mean; //get it
            ground_truth.cov  = submap_gui_target->GetGroundTruth()->covariance; //get it
        }
        else
        {
            ground_truth.mean  = submap_gui_target->GetPose()->GetMean();
            ground_truth.cov(0,0) = sqr(3);
            ground_truth.cov(1,1) = sqr(3);
            ground_truth.cov(2,2) = sqr(D2R(100)); //create a loose one
        }

        if (gui_state==MOVING_CONSTRAINT)
        {
            ground_truth.mean.x() = map_mouse_x;
            ground_truth.mean.y() = map_mouse_y;
            ground_truth.cov(0,0) = sqr(0.5);   //clamp xy cov if moving
            ground_truth.cov(1,1) = sqr(0.5);
        }
        else
        {
            //ROTATING_CONSTRAINT
            //rotate by distance from start: left/right from gt origin
            ground_truth.mean.phi() = gui_rotating_phi_start + 0.1*(ground_truth.mean.x()-map_mouse_x);// - 0.2*(ground_truth.mean.y()-map_mouse_y);
            ground_truth.cov(2,2) = sqr(D2R(15)); //clamp phi cov if rotating
        }

        SubmapGroundTruth *gt = new SubmapGroundTruth( submap_gui_target->GetUUID(),ground_truth.mean, ground_truth.cov );
        mapBuilder->map->SetSubmapGroundTruth( gt );
        SIMPLELOGGER_INFO("CV_EVENT_MOUSEMOVE new ground truth x=%1.1f y=%1.1f phi=%1.1f",ground_truth.mean.x(),ground_truth.mean.y(),R2D(ground_truth.mean.phi()));
    }

    if (event==CV_EVENT_LBUTTONUP && gui_state==MOVING_CONSTRAINT)
    {
        //end of move
        SIMPLELOGGER_INFO("CV_EVENT_LBUTTONUP x=%1.1f y=%1.1f ",map_mouse_x,map_mouse_y);
        gui_state = NONE;
    }

    if (event==CV_EVENT_RBUTTONUP && gui_state==ROTATING_CONSTRAINT)
    {
        //end of rotate
        SIMPLELOGGER_INFO("CV_EVENT_RBUTTONUP x=%1.1f y=%1.1f ",map_mouse_x,map_mouse_y);
        gui_state = NONE;
        //jump cursor to gt
        CPose2D p = submap_gui_target->GetPose()->GetMean();
        if (submap_gui_target->GetGroundTruth()) p = submap_gui_target->GetGroundTruth()->mean;
        int sx = (p.x()-mapBuilder->x_min)* 20.0;
        int sy = (p.y()-mapBuilder->y_max)*-20.0;
        XWarpPointer(display, None, window, 0, 0, 0, 0, sx, sy);
        XFlush(display);
    }
}

void robot1_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid)
{
    grids[1] += grid;
    ROS_INFO("Received robot1 grid (buffer %d)",(uint)grids[1].size());
}

void robot2_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid)
{
    grids[2] += grid;
    ROS_INFO("Received robot2 grid (buffer %d)",(uint)grids[2].size());
}

void robot3_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid)
{
    grids[3] += grid;
    ROS_INFO("Received robot3 grid (buffer %d)",(uint)grids[3].size());
}

void robot4_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid)
{
    grids[4] += grid;
    ROS_INFO("Received robot4 grid (buffer %d)",(uint)grids[4].size());
}

void robot5_gridCallback (const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid)
{
    grids[5] += grid;
    ROS_INFO("Received robot5 grid (buffer %d)",(uint)grids[5].size());
}

void robot1_poseCallback (const geometry_msgs::PoseWithCovarianceStamped & pose)
{
    ROS_INFO("Received robot1 pose=%1.2f,%1.2f",pose.pose.pose.position.x,pose.pose.pose.position.y);
    robot_pose[1] += pose;
}

void robot2_poseCallback (const geometry_msgs::PoseWithCovarianceStamped & pose)
{
    ROS_INFO("Received robot2 pose=%1.2f,%1.2f",pose.pose.pose.position.x,pose.pose.pose.position.y);
    robot_pose[2] += pose;
}

void robot3_poseCallback (const geometry_msgs::PoseWithCovarianceStamped & pose)
{
    ROS_INFO("Received robot3 pose=%1.2f,%1.2f",pose.pose.pose.position.x,pose.pose.pose.position.y);
    robot_pose[3] += pose;
}

void robot4_poseCallback (const geometry_msgs::PoseWithCovarianceStamped & pose)
{
    ROS_INFO("Received robot4 pose=%1.2f,%1.2f",pose.pose.pose.position.x,pose.pose.pose.position.y);
    robot_pose[4] += pose;
}

void robot5_poseCallback (const geometry_msgs::PoseWithCovarianceStamped & pose)
{
    ROS_INFO("Received robot5 pose=%1.2f,%1.2f",pose.pose.pose.position.x,pose.pose.pose.position.y);
    robot_pose[5] += pose;
}

int main(int argc, char **argv)
{
    //ROS subscriptions and advertisement setup
    //calum
    ros::init(argc, argv, "mapbuilder");

    ros::NodeHandle nh("");
    ros::Subscriber robot1_grid_sub_= nh.subscribe("/wambot1/submap", 1, &robot1_gridCallback);
    ros::Subscriber robot1_pose_sub_= nh.subscribe("/wambot1/robot_pose", 1, &robot1_poseCallback);
    ros::Subscriber robot2_grid_sub_= nh.subscribe("/wambot2/submap", 1, &robot2_gridCallback);
    ros::Subscriber robot2_pose_sub_= nh.subscribe("/wambot2/robot_pose", 1, &robot2_poseCallback);
    ros::Subscriber robot3_grid_sub_= nh.subscribe("/wambot3/submap", 1, &robot3_gridCallback);
    ros::Subscriber robot3_pose_sub_= nh.subscribe("/wambot3/robot_pose", 1, &robot3_poseCallback);
    ros::Subscriber robot4_grid_sub_= nh.subscribe("/wambot4/submap", 1, &robot4_gridCallback);
    ros::Subscriber robot4_pose_sub_= nh.subscribe("/wambot4/robot_pose", 1, &robot4_poseCallback);
    ros::Subscriber robot5_grid_sub_= nh.subscribe("/wambot5/submap", 1, &robot5_gridCallback);
    ros::Subscriber robot5_pose_sub_= nh.subscribe("/wambot5/robot_pose", 1, &robot5_poseCallback);
    ros::Publisher map_pub_ = nh.advertise<nav_msgs::OccupancyGrid>("map", 1);
    ros::Publisher robot5_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("wambot5_global_pose", 1);

    tf::TransformBroadcaster tfbr_;
    tf::Transform map_to_robot;

    try
    {
		SIMPLELOGGER_INFO("-------------------------------------------------------------");
		SIMPLELOGGER_INFO("Mapbuilder compiled %s %s",__DATE__,__TIME__);

        cvNamedWindow( "MapBuilder", 1);//CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_NORMAL );// | CV_GUI_EXPANDED );
        cvWaitKey(10);

        if (!(display = XOpenDisplay(NULL)))
        {
            SIMPLELOGGER_ERROR("cannot connect to X server"); exit(1);
        }

        window = (Window)cvGetWindowHandle("MapBuilder");
        int revert=0;
        char *window_name;

        XGetInputFocus(display, &window, &revert);
        XFetchName(display, window, &window_name);

        if (window_name)
        {
            SIMPLELOGGER_INFO("Found window handle: %s", window_name);
        }

        //setup opengl
        GLint attr[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
        XVisualInfo *vi;
        //Window root = DefaultRootWindow(display);
        vi = glXChooseVisual(display, 0, attr);

        if (!(glc = glXCreateContext(display, vi, NULL, GL_TRUE)) )
        {
            SIMPLELOGGER_ERROR("failed to create context");
            exit(1);
        }

        glXMakeCurrent(display, window, glc);

        SIMPLELOGGER_INFO("OpenGL vendor: %s", (const char*)glGetString(GL_VENDOR));

		//global Map holding all submaps, accessed from this thread only
		mapping::Map *map = new mapping::Map();
		MapBuilder *mapBuilder = new MapBuilder( map );
		mapBuilder->Initialize( CONFIG_FILE_NAME );

        cvSetMouseCallback( "MapBuilder", mouse_callback, (void*)mapBuilder);

        char run_id[200];

        char dt_stamp[200];
        time_t t = time(NULL);
        struct tm *tmp = localtime(&t);
        strftime(dt_stamp, sizeof(dt_stamp), "%F_%T", tmp);

        char data_file_name[] = "lab";

        #define	INIT_HEADING M_PI/2
        #define	INIT_X (0)
                //279301.4  + 2100.0/10)
        #define	INIT_Y (0)
                //6130556.0 - 3333.0/10)
        #define INIT_STAGGER false
        #define INIT_MATCH true

        sprintf(run_id,"%s_%s",dt_stamp,data_file_name);
        mapBuilder->run_id = run_id;

        createDirectory("output");

		Submap *previous_submap[MAX_ROBOTS], *current_submap[MAX_ROBOTS];
		CPosePDFGaussian robot_submap_pose[MAX_ROBOTS];
		CPose2D last_slam_pose[MAX_ROBOTS];
		CPose2D submap_pose_local[MAX_ROBOTS];

		//double gps_sum_x[MAX_ROBOTS];
		//double gps_sum_y[MAX_ROBOTS];
		//int gps_count[MAX_ROBOTS];

		double dt_first[MAX_ROBOTS];

        for (int i = 0;i < MAX_ROBOTS;i++)
        {
			previous_submap[i] = NULL;
			current_submap[i] = NULL;

			robot_submap_pose[i] = CPosePDFGaussian();
			last_slam_pose[i]  = CPose2D();
			submap_pose_local[i] = CPose2D();
			dt_first[i] = 0;
		}

		int count_since_finished_data = 0;

        double dt_now = 0;
        TTimeStamp ts_now = now();
        TTimeStamp ts_next_loop = now();
        double total_time_matching=0;

        int map_counter = 0;
        int last_map_counter=-9999;
        char str[222];

        ros::Rate loop_rate(10);

        while(ros::ok())
        {
            loop_rate.sleep();
            ros::spinOnce();
            ts_now = now();

            //inject wambot data here:
            for (int id = 1;id < MAX_ROBOTS;id++)
            {
                //loop over all robots
                do
                {
                    //first delete any poses older than 5 seconds
                    while (robot_pose[id].size())
                    {
                        double dt_pose = (double)robot_pose[id].front().header.stamp.sec + robot_pose[id].front().header.stamp.nsec*1e-9;

                        if (dt_pose>dt_now)
                        {
                            dt_now = dt_pose;
                        }

                        if (dt_now-dt_pose <= 5)
                        {
                            break;
                        }

                        robot_pose[id].pop_front();
                    }

                    if (current_submap[id])
                    {
                        //if we know which submap we're in, find the most recent pose update
                        //with matching submap id
                        UUID current_uuid = current_submap[id]->GetUUID();
                        geometry_msgs::PoseWithCovarianceStamped pose;
                        bool pose_found = false;

                        for (list<geometry_msgs::PoseWithCovarianceStamped>::iterator it= robot_pose[id].begin(); it!=robot_pose[id].end(); )
                        {
                            UUID uuid_pose = it->header.frame_id;

                            if (current_uuid==uuid_pose)
                            {
                                pose = *it;
                                pose_found = true;

                                //delete this pose
                                it = robot_pose[id].erase( it ); //increments iterator
                            }
                            else
                            {
                                it++;
                            }
                        }

                        if (pose_found)
                        {
                            double dt_pose = (double)pose.header.stamp.sec + pose.header.stamp.nsec*1e-9;

                            //pose is ROS geometry_msgs::PoseWithCovarianceStamped
                            //represents pose since last Hector map reset
                            //convert pose orientation quat to yaw theta
                           // double theta = 2*acos( pose.pose.pose.orientation.w );
                            double q_0 = pose.pose.pose.orientation.w;
                            double q_1 = pose.pose.pose.orientation.x;
                            double q_2 = pose.pose.pose.orientation.y;
                            double q_3 = pose.pose.pose.orientation.z;
                            double theta = atan2(2*(q_0*q_3 + q_1*q_2),1 - 2*(q_2*q_2 + q_3*q_3));

                            CPose2D submap_pose = CPose2D(pose.pose.pose.position.x,pose.pose.pose.position.y,theta);

                            //approximate gaussian uncertainty
                            double dist = submap_pose.norm();
                            CMatrixDouble33 cov; cov.zeros();

                            cov(0,0) = dist*sqr(0.6); //wheel slip x
                            cov(1,1) = dist*sqr(0.3); //wheel slip y
                            cov(2,2) = fabs(submap_pose.phi())*sqr(0.05);  //angle 10%

                            robot_submap_pose[id] = CPosePDFGaussian( submap_pose, cov);

                            ROS_INFO("Robot %d, submap:%s, pose: x,y,phi=%1.2f,%1.2f,%1.2f deg has moved %2.2f meters",
                                     id, current_uuid.c_str(), robot_submap_pose[id].mean.x(), robot_submap_pose[id].mean.y(), R2D(robot_submap_pose[id].mean.phi()), dist );

                            //add history:
                            mapBuilder->robot_pose_history[id].push_back(RobotPoseLocal(current_submap[id],robot_submap_pose[id].mean,dt_pose) );
                        }
                    }

                    if (grids[id].size())
                    {
                        //process next grid
                        const boost::shared_ptr<nav_msgs::OccupancyGrid>& grid = grids[id].front();
                        double dt_grid = (double)grid->header.stamp.sec + grid->header.stamp.nsec*1e-9;

                        if (dt_grid>dt_now)
                        {
                            dt_now = dt_grid;
                        }

                        UUID uuid_grid = grid->header.frame_id;
                        previous_submap[id] = current_submap[id];
                        current_submap[id] = map->GetOrCreateSubmap( uuid_grid );

                        if (previous_submap[id] != current_submap[id])
                        {
                            current_submap[id]->robot_id = id;
                            mapBuilder->most_recent_submap = current_submap[id];

                            //we've got a new submap
                            //set initial pose for this new submap
                            CPosePDFGaussian new_submap_pose;

                            if (previous_submap[id] && previous_submap[id]->GetPose())
                            {
                                //new submap UTM pose can be constructed from previous submap and robot pose
                                new_submap_pose = previous_submap[id]->GetPose()->GetPosePDFGaussian() + robot_submap_pose[id]; //robot pose set in previous call
                            }
                            else
                            {
                                //this is the first submap:
                                new_submap_pose.mean.x(   INIT_X  );
                                new_submap_pose.mean.y(   INIT_Y + (INIT_STAGGER?id*1.0:0.0) );
                                new_submap_pose.mean.phi( INIT_HEADING );
                                current_submap[id]->dont_match = !INIT_MATCH;
                                new_submap_pose.cov(0,0) = sqr(300);
                                new_submap_pose.cov(1,1) = sqr(300);
                                new_submap_pose.cov(2,2) = sqr(2*M_PI);
                            }

                            SubmapPose* sp = new SubmapPose( current_submap[id]->GetUUID(), new_submap_pose, false );
                            map->SetSubmapPose( sp );

                            ROS_INFO("New submap %s", current_submap[id]->GetUUID().c_str() );

                            //new submap is linked to a previous submap by a constraint
                            if (previous_submap[id] && previous_submap[id]->GetPose())
                            {
                                //the robot's final pose in the previous submap becomes the constraint:
                                //create submap Constraint object
                                SubmapConstraint *constraint = new SubmapConstraint(	CreateUUID(),
                                                previous_submap[id]->GetUUID(),current_submap[id]->GetUUID(), robot_submap_pose[id].mean,
                                                robot_submap_pose[id].cov, CONSTRAINT_TYPE_ODOMETRY, 0.5 );
                                map->SetSubmapConstraint( constraint );
                            }

                            robot_submap_pose[id] = CPosePDFGaussian();
                        }

                        current_submap[id]->AddOccupancyGridData(dt_grid,grid);
                        grids[id].pop_front();
                    }
                } while (grids[id].size());

            }// end for robot id

            SIMPLELOGGER_INFO("dt_now %1.3f", dt_now);

            bool matching_finished = true;

			//search for submap matches
            if (gui_state==NONE )
            {
                matching_finished = mapBuilder->MatchSubmaps();

                if (matching_finished)
                {
                    total_time_matching = 0.0; //reset counter if finished last match
                }
                else
                {
                    //else add the search time
                    total_time_matching += SUBMAP_MATCHING_TIME_INCREMENT_MAX;
                }

                if (total_time_matching>SUBMAP_MATCHING_TIME_TOTAL_MAX)
                {
                    //if the total searches are more than the allocated search time (0.5sec * num robots)
                    matching_finished = true;
                }
            }

            //sleep if necessary
            if (matching_finished )
            {
                //sleep until next epoch
                while (now()<ts_next_loop) usleep(1000);
            }

            ts_next_loop = now()+(TS_SEC/VIDEO_FRAME_RATE_HZ );

            //process ui input
            char c = (char)cvWaitKey(5);

            if (c==27)
            {
                SIMPLELOGGER_INFO( "Exiting..." );
                break;
            }

            if (c=='t' || c=='T')
            {
                mapBuilder->plot_tracks      = !mapBuilder->plot_tracks;
            }

            if (c=='c' || c=='C')
            {
                mapBuilder->plot_constraints = !mapBuilder->plot_constraints;
            }

            if (c=='^' || c=='_')
            {
                SIMPLELOGGER_INFO( "Core dump..." ); abort();
            }

            if (submap_gui_target && (dist_squared_best < sqr(4.0)))
            {
                CPosePDFGaussian ground_truth;

                gui_rotating_phi_start = (submap_gui_target->GetGroundTruth()) ? submap_gui_target->GetGroundTruth()->mean.phi() : submap_gui_target->GetPose()->GetMean().phi();
                ground_truth.mean.phi() = gui_rotating_phi_start;

                if (submap_gui_target->GetGroundTruth())
                { //already have one
                    ground_truth.mean = submap_gui_target->GetGroundTruth()->mean; //get it
                    ground_truth.cov  = submap_gui_target->GetGroundTruth()->covariance; //get it
                }
                else
                {
                    ground_truth.mean  = submap_gui_target->GetPose()->GetMean();
                    ground_truth.cov(0,0) = sqr(3);
                    ground_truth.cov(1,1) = sqr(3);
                    ground_truth.cov(2,2) = sqr(D2R(100)); //create a loose one
                }

                switch (c)
                {
                    case 'w':
                        ground_truth.mean.y() += 0.05;
                        ground_truth.cov(0,0) = sqr(0.5);   //clamp xy cov if moving
                        ground_truth.cov(1,1) = sqr(0.5);
                    break;

                    case 'a':
                        ground_truth.mean.x() -= 0.05;
                        ground_truth.cov(0,0) = sqr(0.5);   //clamp xy cov if moving
                        ground_truth.cov(1,1) = sqr(0.5);
                    break;

                    case 's':
                        ground_truth.mean.y() -= 0.05;
                        ground_truth.cov(0,0) = sqr(0.5);   //clamp xy cov if moving
                        ground_truth.cov(1,1) = sqr(0.5);
                    break;

                    case 'd':
                        ground_truth.mean.x() += 0.05;
                        ground_truth.cov(0,0) = sqr(0.5);   //clamp xy cov if moving
                        ground_truth.cov(1,1) = sqr(0.5);
                    break;

                    case 'q':
                        //ROTATING_CONSTRAINT
                        //rotate by distance from start: left/right from gt origin
                        ground_truth.mean.phi() += D2R(1);
                        ground_truth.cov(2,2) = sqr(D2R(15)); //clamp phi cov if rotating
                    break;

                    case 'e':
                        //ROTATING_CONSTRAINT
                        //rotate by distance from start: left/right from gt origin
                        ground_truth.mean.phi() -= D2R(1);
                        ground_truth.cov(2,2) = sqr(D2R(15)); //clamp phi cov if rotating
                    break;
                }

                SubmapGroundTruth *gt = new SubmapGroundTruth( submap_gui_target->GetUUID(),ground_truth.mean, ground_truth.cov );
                mapBuilder->map->SetSubmapGroundTruth( gt );
                SIMPLELOGGER_INFO("CV_EVENT_KEYBOARD new ground truth x=%1.1f y=%1.1f phi=%1.1f",ground_truth.mean.x(),ground_truth.mean.y(),R2D(ground_truth.mean.phi()));
            }

            //do the optimisation
            mapBuilder->OptimiseSubmaps();//must opt after matching so stats are valid

            bool save_frame = true;//matching_finished;

            mapBuilder->CreateNewMapUUID();
            mapBuilder->submap_highlight = gui_state ? submap_gui_target : 0;
            mapBuilder->BuildMap( dt_now, save_frame );

            map_pub_.publish(mapBuilder->ros_map);
            ROS_INFO("published ROS map");

            //ROS publish the postion of the latest submap for each robot.
            ////////////////////////////////////////////////////////
            for (int id=0; id<MAX_ROBOTS; id++)
            {
                if (current_submap[id])
                {
                    CPose2D p = current_submap[id]->GetPose()->GetMean() + robot_submap_pose[id].mean;

                    tf_map_to_robot.setOrigin( tf::Vector3( p.x(), p.y(), 0) );
                    tf_map_to_robot.setRotation( tf::createQuaternionFromRPY(0, 0, p.phi()));

                    //tf::Quaternion(0,0,sin(p.phi()/2), cos(zp.phi()/2)));

                    static char robotX_base_link[255]  = "robot5_base_link";
                    static char robotX_base_laser[255] = "robot5_base_laser";

                    sprintf(robotX_base_link, "robot%d_base_link",id);
                    sprintf(robotX_base_laser,"robot%d_base_laser",id);

                    tfbr_.sendTransform(tf::StampedTransform(tf_map_to_robot,ros::Time::now(),"map",robotX_base_link));
                    tfbr_.sendTransform(tf::StampedTransform(tf::Transform(tf::Quaternion(0, 0, 0, 1), tf::Vector3(0.1, 0.0, 0.0)),ros::Time::now(),robotX_base_link, robotX_base_laser));
                }
            }

            if (save_frame)
            {
                mapBuilder->stats["dt"] = dt_now;
                mapBuilder->stats["map frame counter"] = map_counter++;
                mapBuilder->stats_data.push_back( mapBuilder->stats );

                for (std::map<string,double>::iterator it = mapBuilder->stats.begin(); it!= mapBuilder->stats.end(); it++ )
                {
                    mapBuilder->stats_header.insert( it->first ); //make sure this item is in the header
                    //end of cycle, setup for gathering next line of statistics:
                }
            }

            mapBuilder->stats.clear();
		} //end while

        SIMPLELOGGER_INFO( "Dumping position data..." );

        //dump robot positions
        for (int i=0; i<MAX_ROBOTS; i++)
        {
            if (mapBuilder->robot_pose_history[i].size()>1)
            {
                char filename[222];
                sprintf(filename,"output/%s_robot_%d_track.csv",run_id,i);

                FILE *f = fopen(filename,"wt");
                BOOST_FOREACH(RobotPoseLocal r, mapBuilder->robot_pose_history[i] )
                {
                    CPose2D p = r.submap->GetPose()->GetMean() + r.pose;
                    fprintf(f,"%d,%1.1f,%1.1f,%1.3f\n",r.dt, p.x(), p.y(),p.phi());
                }

                fclose(f);
            }
        }

        SIMPLELOGGER_INFO( "Complete." );

        //dump stats data to file
        if (mapBuilder->stats_header.size())
        {
            SIMPLELOGGER_INFO( "Dumping stats..." );
            char filename[222];

            sprintf(filename,"output/%s_stats.csv",run_id );
            FILE *f = fopen(filename,"wt");

            for ( set<string>::iterator ii = mapBuilder->stats_header.begin(); ii!=mapBuilder->stats_header.end(); ii++ )
            {
                fprintf(f,"%s,",ii->c_str()); //print header for element ii
            }

            fprintf(f,"\n");

            for ( list<std::map<string,double> >::iterator jj = mapBuilder->stats_data.begin(); jj!=mapBuilder->stats_data.end(); jj++ )
            {
                for ( set<string>::iterator ii = mapBuilder->stats_header.begin(); ii!=mapBuilder->stats_header.end(); ii++ )
                {
                    fprintf(f,"%g,",(*jj)[*ii]); //print data for row jj, element ii
                }

                fprintf(f,"\n");
            }

            fclose(f);
            SIMPLELOGGER_INFO( "Complete." );
        }

        if (map)
        {
            delete map;
        }

        if (mapBuilder)
        {
            delete mapBuilder;
        }

    }
    catch (std::exception &event)
    {
		std::cerr << event.what() << std::endl << "Exception" << std::endl;
		return -1;
    }
    catch (...)
    {
		std::cerr << "Untyped exception" << std::endl;
		return -1;
	}

    glXMakeCurrent( display, None, NULL );
    glXDestroyContext( display, glc );

	SIMPLELOGGER_INFO("Exit.");
	return 0;
}
