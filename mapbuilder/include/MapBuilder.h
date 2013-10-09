#include <mrpt/base.h>
#include <mrpt/slam.h>
#include <mrpt/gui.h>

//#include "DdsFactory.h"

#include "Mapping.h"
#include "Utm.h"
#include "simplelog.h"

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <Eigen/Dense>
#include "spa2d.h"

#define TS_SEC (10000000) //10000000
#define VIDEO_FRAME_RATE_HZ (10)
#define COST_INCREASE_THRESHOLD (0.03)
#define COST_MINIMUM_THRESHOLD  (0.01)
#define MAX_ROBOTS (100)
#define SUBMAP_MATCHING_TIME_TOTAL_MAX (5)     //5
#define SUBMAP_MATCHING_TIME_INCREMENT_MAX (0.1)
#define MATCH_REL_POSE_THRESH (15.0)
#define MATCH_SEARCH_WIDTH (2* 5.0)     //5
#define MATCH_SEARCH_ANGLE (D2R(2*20.0)) //20
#define SINGLE_MATCH_COUNT (3) //Use this in DoSingleMatch to eliminate bingo numbers (3)
//#define WRITE_MATCHING 1

typedef boost::shared_ptr<nav_msgs::OccupancyGrid> GridPtr;

enum playback_state_types
{
    FASTEST,
    REALTIME,
    PAUSED
};

enum gui_state_types
{
    NONE,
    MOVING_CONSTRAINT,
    ROTATING_CONSTRAINT
};

using namespace mrpt::system;
using namespace sba;

namespace mapping
{

    struct SlamPoseLocal
    {
        string vehicle_name;
        string submap_uuid; /* maximum length = (255) */
        CPose2D robot_pose;
    };

    struct RobotPoseLocal
    {
        Submap *submap;
        CPose2D pose;
        int dt;
        public: RobotPoseLocal( Submap *_submap, CPose2D _pose, int _dt) : submap(_submap), pose(_pose), dt(_dt) {}
    };

    class MapBuilder
    {
    public:
        //main map object:
        Map *map;
        Submap *most_recent_submap;

        //count mapping steps
        int step;

        char config_filepath[255];    
        char submaps_path[255];
        char maps_path[255];
        char matches_path[255];
        char map_state_path[255];

        CvFont font,font_small;
        FILE *f_encoder;
        IplImage* ground_truth_img;
        IplImage* map_img;
        IplImage* composite_img;

        std::list<double> submap_scores;

      public:
        //ROS map output data type
        //nav_msgs::OccupancyGrid ros_map;
        GridPtr ros_map;

        bool plot_tracks;
        bool plot_constraints;
        double x_min;
        double y_min;
        double x_max;
        double y_max;

        set<string> stats_header;
        list<std::map<string,double> > stats_data;
        std::map<string,double> stats;

        Submap* submap_highlight;

        char * run_id;

        std::vector<CvScalar> robot_color_rgb;
        std::list<Submap*>::reverse_iterator matching_submap_iterator;

        //opengl stuff:
        //HWND hWnd;
        GLuint submap_blend_fs;
        GLuint submap_blend_vs;
        GLuint submap_blend_sp;
        GLuint submap_matcher_fs;
        GLuint submap_matcher_vs;
        GLuint submap_matcher_sp;

        //uniforms:
        GLuint submap_matcher_scale;
        GLuint submap_matcher_base_bottom_left;
        GLuint submap_matcher_base_top_right;
        GLuint submap_matcher_target_bottom_left;
        GLuint submap_matcher_target_top_right;
        GLuint submap_matcher_initial_offset;
        GLuint submap_matcher_base_submap_tex;
        GLuint submap_matcher_target_submap_tex;

        GLuint submap_blend_tex;

        //map of textures, one per submap
        std::map<UUID,GLuint*> occupancy_texture;

        //render buffers
        GLuint map_framebuffer;
        GLuint map_renderbuffers[2];
        GLuint matching_framebuffer;
        GLuint matching_renderbuffer;

        int map_pixel_width;
        int map_pixel_height;

        int matching_pixel_width;
        int matching_pixel_height;

        //configuration file parameters:
        int generatedMapSizeMultiple;
        int submapMatchingMinimumOccupiedCellCount;

        bool CheckOpenGlError( char *extra = "");

        string GENERATED_MAP_FORMAT;

        char utm_zone[5];
        double REFERENCE_POSITION_LATITUDE;
        double REFERENCE_POSITION_LONGITUDE;
        CPose2D reference_position;

        double SUBMAP_POSE_CORRECTION_MIN_LIN_DIST;
        double SUBMAP_POSE_CORRECTION_MIN_ANG_DIST;

        double HMI_GROUND_TRUTH_CONSTRAINT_ERROR_LIN_STDEV;
        double HMI_GROUND_TRUTH_CONSTRAINT_ERROR_ANG_STDEV;

        CPose2D *robot_pose;
        Submap *current_submap;

        typedef std::pair<Submap*,Submap*> SubmapPair;
        std::map<SubmapPair,CPose2D> submap_submap_match_tested;

        mapping::UUID generated_map_uuid;

    public:
        int counter;
        double submapMatchingScoreThreshold;
        double submapMatchingScoreThresholdWithPrior;


        MapBuilder(Map *_map) :
            map ( _map ),
            ros_map(new nav_msgs::OccupancyGrid())
        {
            submap_blend_fs = 0;
            submap_blend_vs = 0;
            submap_blend_sp = 0;
            submap_matcher_fs = 0;
            submap_matcher_vs = 0;
            submap_matcher_sp = 0;
            //version = 0;
            robot_pose = 0;
            current_submap = 0;
            generated_map_uuid = CreateUUID();
            counter = 0;

            x_min = 0.0;
            x_max = 0.0;
            y_min = 0.0;
            y_max = 0.0;
        };

        ~MapBuilder(void);

        void Initialize( char *config_filename );

        void SetRobotPose( SlamPoseLocal *pose )
        {
            Submap *submap = map->GetSubmap( pose->submap_uuid );

            if (submap && submap->GetPose())
            {
                //this submap is known about, set the robot's pose within it
                //any skipped poses should be repeated 1/5 second later once submap pose received.
                if (robot_pose)
                {
                    delete robot_pose;
                }

                robot_pose = new CPose2D( pose->robot_pose );
                current_submap = submap;
                //SIMPLELOGGER_DEBUG("New SlamPose, %s submap %s, %1.1f, %1.1f, %1.2f deg.", pose->vehicle_name.c_str(), submap->GetUUID().substr(0,6).c_str(), robot_pose->x(), robot_pose->y(), R2D(robot_pose->phi()) );
            }
            else
            {
                SIMPLELOGGER_DEBUG("New SlamPose, %s unknown submap", pose->vehicle_name.c_str() );
            }
            delete pose;
        }

        bool OptimiseSubmaps( SubmapConstraint* test_constraint = NULL );
        double OptimiseLocalConstraintCost( SysSPA2d &spa, SubmapConstraint* test_constraint );
        bool BuildMap( int dt_rel_last,  bool save_frame );
        bool WritePersistent();
        bool ReadPersistent( char * data_filepath );
        bool MatchSubmaps();
        double DoSingleMatch( float x_range, float y_range, float phi_range, CPose2D *initial_offset, CPose2D *measured_offset, Eigen::Matrix3d *measured_cov, Submap* base_submap,Submap* target_submap, SubmapConstraint* constraint  );

        void CreateNewMapUUID()
        {
            char temp[255];
            sprintf(temp, "%08d",counter++ );
            generated_map_uuid = temp;
        }

        public:	list<RobotPoseLocal> robot_pose_history[MAX_ROBOTS];
    };
}
